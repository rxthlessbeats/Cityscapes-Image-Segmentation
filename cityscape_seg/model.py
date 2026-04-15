"""FCN-8s model built from scratch for semantic segmentation.

Architecture following Long et al. "Fully Convolutional Networks
for Semantic Segmentation" (CVPR 2015):

  Encoder : 5 VGG-style blocks (conv-bn-relu pairs + maxpool)
  Bridge  : two 1x1 conv layers (replace the original FC6/FC7)
  Decoder : transposed convolutions with FCN-8s skip connections
            from pool3 and pool4 for fine-grained spatial detail
"""

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


# ---------------------------------------------------------------------------
# Model registry -- add new models here
# ---------------------------------------------------------------------------
MODEL_REGISTRY: dict[str, type[nn.Module]] = {
    "fcn8s": FCN8s,
    "unet": UNet,
}


def build_model(config: TrainConfig) -> nn.Module:
    """Instantiate the model specified by ``config.model_name``."""
    cls = MODEL_REGISTRY[config.model_name]
    return cls(config.num_classes)
