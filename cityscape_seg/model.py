"""Segmentation models built from scratch: FCN-8s, U-Net, DeepLabV3+."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from .config import TrainConfig


class ConvBlock(nn.Module):
    """N consecutive (Conv3x3 -> BN -> ReLU) layers."""

    def __init__(self, in_ch: int, out_ch: int, n_convs: int = 2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(n_convs):
            layers.append(nn.Conv2d(in_ch if i == 0 else out_ch, out_ch, 3, padding=1))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class FCN8s(nn.Module):
    """FCN-8s (Long et al., 2015) built from scratch."""

    def __init__(self, num_classes: int) -> None:
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(3, 64, n_convs=2)
        self.enc2 = ConvBlock(64, 128, n_convs=2)
        self.enc3 = ConvBlock(128, 256, n_convs=3)
        self.enc4 = ConvBlock(256, 512, n_convs=3)
        self.enc5 = ConvBlock(512, 512, n_convs=3)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bridge (replaces FC6 / FC7)
        self.bridge = nn.Sequential(
            nn.Conv2d(512, 1024, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
            nn.Conv2d(1024, 1024, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.5),
        )

        # Decoder (FCN-8s)
        self.score_bridge = nn.Conv2d(1024, num_classes, 1)

        self.up_bridge = nn.ConvTranspose2d(
            num_classes,
            num_classes,
            4,
            stride=2,
            padding=1,
        )
        self.score_pool4 = nn.Conv2d(512, num_classes, 1)

        self.up_pool4 = nn.ConvTranspose2d(
            num_classes,
            num_classes,
            4,
            stride=2,
            padding=1,
        )
        self.score_pool3 = nn.Conv2d(256, num_classes, 1)

        self.up_final = nn.ConvTranspose2d(
            num_classes,
            num_classes,
            16,
            stride=8,
            padding=4,
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool(e1)

        e2 = self.enc2(p1)
        p2 = self.pool(e2)

        e3 = self.enc3(p2)
        p3 = self.pool(e3)  # pool3 skip

        e4 = self.enc4(p3)
        p4 = self.pool(e4)  # pool4 skip

        e5 = self.enc5(p4)
        p5 = self.pool(e5)

        # Bridge
        b = self.bridge(p5)

        # Decoder with skip connections
        s_bridge = self.score_bridge(b)
        up1 = self.up_bridge(s_bridge)
        s_pool4 = self.score_pool4(p4)
        fuse1 = up1 + s_pool4

        up2 = self.up_pool4(fuse1)
        s_pool3 = self.score_pool3(p3)
        fuse2 = up2 + s_pool3

        out = self.up_final(fuse2)
        return out


class UNet(nn.Module):
    """U-Net (Ronneberger et al., 2015) for semantic segmentation.

    Symmetric encoder-decoder with skip connections via concatenation.
    """

    def __init__(self, num_classes: int, base_ch: int = 32) -> None:
        super().__init__()

        self.enc1 = ConvBlock(3, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)
        self.enc4 = ConvBlock(base_ch * 4, base_ch * 8)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(base_ch * 8, base_ch * 16)

        self.up4 = nn.ConvTranspose2d(base_ch * 16, base_ch * 8, 2, stride=2)
        self.dec4 = ConvBlock(base_ch * 16, base_ch * 8)

        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec3 = ConvBlock(base_ch * 8, base_ch * 4)

        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_ch * 4, base_ch * 2)

        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = ConvBlock(base_ch * 2, base_ch)

        self.head = nn.Conv2d(base_ch, num_classes, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.head(d1)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling (Chen et al., 2017).

    Parallel branches with different dilation rates capture multi-scale context.
    """

    def __init__(self, in_ch: int, out_ch: int = 256, rates: tuple[int, ...] = (6, 12, 18)) -> None:
        super().__init__()
        modules: list[nn.Module] = [
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        ]
        for rate in rates:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, padding=rate, dilation=rate, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )
            )
        # Image-level features (global average pooling; no BN -- spatial size is 1x1)
        modules.append(
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_ch, out_ch, 1),
                nn.ReLU(inplace=True),
            )
        )
        self.branches = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * (len(rates) + 2), out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        results = []
        for branch in self.branches[:-1]:
            results.append(branch(x))
        img_feat = self.branches[-1](x)
        img_feat = nn.functional.interpolate(
            img_feat, size=x.shape[2:], mode="bilinear", align_corners=False
        )
        results.append(img_feat)
        return self.project(torch.cat(results, dim=1))


class DeepLabV3Plus(nn.Module):
    """DeepLabV3+ (Chen et al., 2018) built from scratch.

    Encoder : 4 VGG-style stages (ConvBlock + MaxPool), output stride 16
    ASPP    : multi-scale context at the encoder output
    Decoder : low-level skip from stage 1 (stride 4), refine + upsample
    """

    def __init__(self, num_classes: int, base_ch: int = 32) -> None:
        super().__init__()
        self.enc1 = ConvBlock(3, base_ch, n_convs=2)
        self.enc2 = ConvBlock(base_ch, base_ch * 2, n_convs=2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4, n_convs=3)
        self.enc4 = ConvBlock(base_ch * 4, base_ch * 8, n_convs=3)
        self.pool = nn.MaxPool2d(2)

        aspp_out_ch = 256
        self.aspp = ASPP(base_ch * 8, out_ch=aspp_out_ch)

        self.low_level_proj = nn.Sequential(
            nn.Conv2d(base_ch, 48, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(aspp_out_ch + 48, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.head = nn.Conv2d(256, num_classes, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        h, w = x.shape[2:]

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        deep = self.pool(e4)

        aspp_out = self.aspp(deep)

        low_level = self.low_level_proj(self.pool(e1))

        aspp_up = nn.functional.interpolate(
            aspp_out, size=low_level.shape[2:], mode="bilinear", align_corners=False
        )
        fused = torch.cat([aspp_up, low_level], dim=1)
        refined = self.refine(fused)

        out = self.head(refined)
        out = nn.functional.interpolate(out, size=(h, w), mode="bilinear", align_corners=False)
        return out


# ---------------------------------------------------------------------------
# Model registry -- add new models here
# ---------------------------------------------------------------------------
MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "fcn8s": FCN8s,
    "unet": UNet,
    "deeplabv3plus": DeepLabV3Plus,
}


def build_model(config: TrainConfig) -> nn.Module:
    """Instantiate the model specified by ``config.model_name``."""
    cls = MODEL_REGISTRY[config.model_name]
    return cls(config.num_classes)
