# Cityscapes Semantic Segmentation

FCN-8s semantic segmentation on the Cityscapes dataset, with an 8-class label scheme.

## Project Structure

```
cityscape_seg/
  .env                  # machine / environment settings (data path, device, workers)
  config.yaml           # training hyperparameters
  requirements.txt      # Python dependencies
  data/                 # dataset (not tracked in git)
    gtFine/             # raw Cityscapes annotations
    leftImg8bit/        # raw Cityscapes images
    small_data/         # flattened split used for training
      train/
      valid/
      test/
    create_data.ipynb   # script to build small_data/ from the raw downloads
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

### 3. Prepare the dataset

The model trains on a flattened version of Cityscapes stored under `data/small_data/`.
A notebook at `data/create_data.ipynb` automates this.

1. Download the two Cityscapes archives (requires a free account at [cityscapes-dataset.com](https://www.cityscapes-dataset.com/)):
   - **gtFine** (annotations): [packageID=1](https://www.cityscapes-dataset.com/file-handling/?packageID=1)
   - **leftImg8bit** (images): [packageID=3](https://www.cityscapes-dataset.com/file-handling/?packageID=3)
2. Extract them so the `data/` folder looks like this:
   ```
   data/
     gtFine/
       train/
       val/
       test/
     leftImg8bit/
       train/
       val/
       test/
   ```
3. Open `data/create_data.ipynb` and run all cells. It copies `*_labelIds.png` and `*_leftImg8bit.png` files from every city into flat directories:
   ```
   data/small_data/
     train/    (all training cities)
     valid/    (all validation cities)
     test/     (all test cities)
   ```

After this step, `data/small_data/` is ready and the training pipeline can load it.

### 4. Configure environment

Edit `.env` to match your machine:

```dotenv
CITYSEG_DATA_ROOT=./data/small_data
CITYSEG_DEVICE=cuda          # or "cpu"
CITYSEG_NUM_WORKERS=0
CITYSEG_PIN_MEMORY=true
```

### 5. Configure training

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

### View TensorBoard logs

Each training run (both CLI and notebook) automatically logs metrics to TensorBoard under the `runs/` directory. To launch the dashboard:

```bash
tensorboard --logdir runs
```

**What gets logged:**

- **Per-epoch scalars** -- train/val loss, train/val accuracy, learning rate
- **Per-class IoU and mIoU** -- logged at the end of training
- **Prediction images** -- input / ground-truth / prediction grid
- **Hyperparameters** -- batch size, lr, weight decay, loss type, etc. (viewable in the HParams tab for comparing runs)

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
