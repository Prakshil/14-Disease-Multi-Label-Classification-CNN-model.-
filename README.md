# NIH Chest X-Ray 14-Disease Multi-Label Classification

A research-grade deep learning pipeline for multi-label classification of 14 thoracic diseases from chest X-ray images. Implements DenseNet121, EfficientNetB0, and a Custom CNN with a comprehensive training and evaluation workflow.

## Project Overview

This project uses the NIH Chest X-Ray 14 dataset (224x224 resized) to classify 14 common chest pathologies:

- Atelectasis, Cardiomegaly, Consolidation, Edema, Effusion, Emphysema, Fibrosis, Hernia, Infiltration, Mass, Nodule, Pleural_Thickening, Pneumonia, Pneumothorax

### Key Features

- **Multi-Label Classification**: Handles co-occurring diseases via multi-hot encoding and weighted BCE loss
- **Two-Phase Training**: Frozen backbone pretraining followed by full fine-tuning
- **Data Leakage Prevention**: MultilabelStratifiedShuffleSplit for robust train/val/test splitting
- **Dynamic Preprocessing**: CLAHE enhancement, real-time augmentation (rotation, affine transforms)
- **Mixed Precision Training**: Uses PyTorch AMP (`GradScaler` + `autocast`) for faster GPU training
- **AUC-Weighted Ensemble**: Dynamically weights model contributions based on validation AUC
- **GradCAM Visualization**: Patched DenseNet for in-place ReLU-compatible saliency maps

## Architecture

Three model architectures are supported:

| Model | Backbone | Params | Notes |
|-------|----------|--------|-------|
| DenseNet121 | `densenet121` ~7M pretrained | + 512→256→128→14 head | Primary model |
| EfficientNetB0 | `efficientnet_b0` ~4M pretrained | + 512→256→128→14 head | Secondary model |
| Custom CNN | Fully scratch | Configurable | Baseline comparison |

All models output 14 logits (one per disease class) with `BCEWithLogitsLoss`.

## Setup

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended) or CPU
- Kaggle API access (for dataset download)

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd nih-chestxray-multilabel

# Create virtual environment
python -m venv .venv

# Activate it
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\Activate.ps1 # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dataset

The dataset is downloaded automatically via `kagglehub` on first run:
- Source: [khanfashee/nih-chest-x-ray-14-224x224-resized](https://www.kaggle.com/datasets/khanfashee/nih-chest-x-ray-14-224x224-resized)
- Cached at: `~/.cache/kagglehub/datasets/...`

## Web App (Streamlit)

A standalone Streamlit UI is included for interactive inference:

```bash
streamlit run app.py
```

Features:
- Model selection (DenseNet Frozen, DenseNet Finetuned, Ensemble)
- Upload chest X-ray (PNG/JPG)
- Per-disease probability scores with bar chart
- Disease descriptions, common causes, and radiographic findings
- Adjustable decision threshold
- Performance metrics display

## Usage (Notebook)

Open the Jupyter notebook and run cells sequentially:

```bash
jupyter notebook multi_chestxray.ipynb
```

Or run headlessly:

```bash
jupyter nbconvert --to script multi_chestxray.ipynb --stdout | python
```

### Pipeline Stages

1. **Data Preparation** - Download, parse CSV labels, multi-hot encoding
2. **Configuration** - Hyperparameters, class weights, dataset splits
3. **Data Generators** - CLAHE preprocessing, augmentation pipeline, DataLoaders
4. **Model Architecture** - DenseNet121/EfficientNetB0 with custom classifier head
5. **Training (Phase A)** - Frozen backbone, ~0.001 LR, 10 epochs
6. **Training (Phase B)** - Full fine-tuning, ~1e-5 LR, 10 epochs
7. **Evaluation** - Per-class thresholds, F1, AUROC, sensitivity/specificity
8. **Ensemble** - AUC-weighted model combination
9. **GradCAM** - Saliency map visualization for interpretability

## Results

| Metric | DenseNet Frozen | DenseNet Finetune | Ensemble |
|--------|----------------|-------------------|----------|
| Macro F1 | — | — | 0.2816 |
| Micro F1 | — | — | 0.3498 |
| Avg AUROC | — | ~0.812 | — |

Per-class performance metrics are displayed in the notebook evaluation cells.

## Project Structure

```
├── multi_chestxray.ipynb   # Main pipeline notebook
├── requirements.txt         # Python dependencies
├── .gitignore
├── README.md
├── output/                  # Saved model weights (generated)
│   ├── best_densenet_frozen.pth
│   └── best_densenet_finetuned.pth
└── .venv/                   # Virtual environment (not tracked)
```

## License

This project is for educational and research purposes. The NIH Chest X-Ray dataset is public domain. Model weights and code are provided as-is.
