# NIH Chest X-Ray 14-Disease Multi-Label Classification

A research-grade deep learning pipeline for multi-label classification of 14 thoracic diseases from chest X-ray images. Designed specifically to tackle long-tailed disease distributions, label dependencies, and cross-dataset domain generalization.

## Project Overview

This project uses the NIH ChestX-ray14 dataset (112,120 frontal-view X-rays from 30,805 unique patients) to classify 14 common chest pathologies. It moves beyond standard CNN baselines by incorporating modern foundation model initializations, graph neural networks for label correlation, and rigorous clinical safety metrics (calibration and abstention).

### Key Features

- **Strict Patient-Wise Splitting**: Zero data leakage guaranteed via `MultilabelStratifiedShuffleSplit` grouped by patient ID.
- **Advanced Encoders**: Supports DenseNet121, EfficientNet-B0, and **ConvNeXt-Large**.
- **Foundation Model Initializations**: Ablation support for standard ImageNet, **DINOv2-Large** (self-supervised), and **MedCLIP** (BiomedVLP-CXR-BERT vision encoder).
- **Imbalance-Aware Losses**: Supports Weighted BCE, Asymmetric Loss (ASL), and **LDAM-BCE with Deferred Re-Weighting (DRW)** to drastically improve rare-class recall.
- **Label Graph Dependency Module**: A residual **2-layer GCNConv** (via `torch_geometric`) that models disease co-occurrences.
- **Clinical Safety & Uncertainty**: 
  - Post-hoc Isotonic Regression Calibration (ECE & Brier Scores).
  - **MC Dropout Uncertainty Estimation** with abstention logic (rejecting the 5% most uncertain predictions).
- **Domain Generalization**: Built-in zero-shot transfer evaluation loops for **CheXpert** and **MIMIC-CXR** datasets with strict label mapping.
- **Multi-GPU Ready**: Automatically detects and utilizes `torch.nn.DataParallel` for cloud compute (e.g., Kaggle T4x2).

## Installation

### Prerequisites
- Python 3.10+
- CUDA-capable GPU (Highly Recommended for ConvNeXt-Large)

### Setup Environment
```bash
# Clone the repository
git clone <repo-url>
cd nih-chestxray-multilabel

# Create virtual environment
python -m venv .venv

# Activate it
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\Activate.ps1 # Windows

# Install core dependencies
pip install pandas numpy matplotlib seaborn scikit-learn opencv-python iterative-stratification
pip install kagglehub datasets transformers tqdm

# Install PyTorch and Torch Geometric (Match to your CUDA version)
# Example for PyTorch 2.0+ and CUDA 12.1:
pip install torch torchvision
pip install torch_geometric
```

## Dataset Handling

The NIH ChestX-ray14 dataset is handled automatically. On the first run of the notebook, `kagglehub` will download the dataset (`khanfashee/nih-chest-x-ray-14-224x224-resized`) directly to your machine's cache and parse the paths.

External validation sets (**CheXpert** and **MIMIC-CXR**) must be downloaded manually from their respective sources (Stanford ML Group and PhysioNet) and their local paths inserted into Phase 18 of the notebook.

## Execution Pipeline

The pipeline is organized into a linear 19-Phase Jupyter Notebook (`multi_chestxray.ipynb`), following best practices for medical AI publications:

1. **Phase 1-4: Data Engineering** - Download, multi-hot encoding, patient-level stratified splitting, class imbalance tracking.
2. **Phase 5-7: Architecture Selection** - Define Loss Functions (W-BCE, ASL, LDAM), build the MultiLabelModel (Backbone + Init + GCNConv), prepare CLAHE DataLoaders.
3. **Phase 8-9: Two-Phase Training** - 
   - *Phase A*: Train custom classifier head with a frozen backbone (LR 1e-3).
   - *Phase B*: Fine-tune the entire network (LR 1e-5) and activate LDAM-DRW class weights at Epoch 5. 
   - *Checkpoints are auto-resumed on failure.*
4. **Phase 10-13: Calibration & Statistics** - Isotonic Regression, Threshold Optimization, Bootstrap 95% Confidence Intervals, and McNemar significance testing.
5. **Phase 14-17: Safety & Interpretability** - Error Analysis (Top FP/FNs), MC Dropout Abstention, Robustness Benchmarking (Blur, JPEG, Brightness), and custom GradCAM visualization.
6. **Phase 18-19: External Validation** - Zero-shot testing on CheXpert and MIMIC-CXR, generating the final cross-dataset Domain Shift tables.

## Reproducibility

For a rigorous paper submission, the notebook includes:
- **Appendix C**: Inference speed benchmarking (ms/image, FPS, Parameter count).
- **Appendix D**: Seed Reproducibility scaffolding to report `mean ± std` across three random seeds (e.g., 42, 123, 999).

## License

This project is for educational and research purposes. The datasets (NIH, CheXpert, MIMIC-CXR) are governed by their respective licenses and data use agreements. Model weights and code are provided as-is.
