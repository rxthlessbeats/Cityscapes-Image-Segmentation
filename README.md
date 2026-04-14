# Cityscapes Semantic Segmentation

FCN-8s semantic segmentation on the Cityscapes dataset, with an 8-class label scheme.

## Project Structure

```
cityscape_seg/
  .env                  # machine / environment settings (data path, device, workers)
  config.yaml           # training hyperparameters
  requirements.txt      # Python dependencies
  data/                 # dataset (not tracked in git)
    small_data/
      train/
      valid/
  cityscape_seg/        # Python package
    config.py           # Pydantic Settings + TrainConfig
    labels.py           # class names, colours, label remap table
    transforms.py       # paired image-label augmentations
    dataset.py          # CityscapesSegDataset
    model.py            # ConvBlock, FCN8s
    loss.py             # FocalLoss, build_criterion()
    train.py            # training / validation loops
    evaluate.py         # mIoU, qualitative visualisation
    utils.py            # inv_normalize, label_to_color
    cli.py              # CLI entry point
    __main__.py          # python -m support
  notebooks/
    base.ipynb          # interactive exploration notebook
```

## Setup

### 1. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

Edit `.env` to match your machine:

```dotenv
CITYSEG_DATA_ROOT=./data/small_data
CITYSEG_DEVICE=cuda          # or "cpu"
CITYSEG_NUM_WORKERS=0
CITYSEG_PIN_MEMORY=true
```

### 4. Configure training

Edit `config.yaml` to adjust hyperparameters:

```yaml
img_height: 256
img_width: 512
batch_size: 4
num_classes: 8
num_epochs: 30
lr: 0.0001
weight_decay: 0.001
num_train: 400
num_val: 100
seed: 42
loss_type: cross_entropy   # or "focal"
focal_gamma: 2.0
```

## Usage

### Train via CLI

From the project root:

```bash
python -m cityscape_seg train
```

To use a custom config file:

```bash
python -m cityscape_seg train --config my_config.yaml
```

### Use the notebook

Open `notebooks/base.ipynb` in Jupyter or VS Code. The first cell adds the project root to `sys.path` so all package imports work without installation.

## 8-Class Label Scheme

| ID | Class          | Cityscapes labelIds               |
|----|----------------|-----------------------------------|
| 0  | road/drivable  | 7, 9, 10                          |
| 1  | sidewalk       | 8                                 |
| 2  | human          | 24, 25                            |
| 3  | vehicle        | 26-33                             |
| 4  | traffic object | 17-20                             |
| 5  | nature         | 21-23 (vegetation, terrain, sky)  |
| 6  | construction   | 11-16 (building, wall, fence ...) |
| 7  | background     | everything else (void)            |

## Model

Custom FCN-8s following Long et al. (CVPR 2015):

- **Encoder**: 5 VGG-style blocks with BatchNorm
- **Bridge**: two 1x1 conv layers with dropout (replaces FC6/FC7)
- **Decoder**: transposed convolutions with skip connections from pool3 and pool4
- ~16.3M parameters
