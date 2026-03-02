# Temporal Bone HRCT - AI Pathology Detection

AI-based detection of middle ear pathologies from HRCT temporal bone scans using 3D ResNet-18 with CBAM attention.

**Pathologies Detected:**

- Cholesteatoma
- Ossicular chain discontinuity
- Facial nerve dehiscence (production)

---

## Project Structure

```
HRCT_Temporal/
├── dicom_input/              # Raw DICOM files
├── processed_data/           # Preprocessed volumes (Phase 1)
├── landmarks_detected/       # Detected 3D landmarks (Phase 2A)
├── roi_extracted/            # Final Middle Ear ROIs (Phase 2B)
├── dataset_splits/           # Train/Val/Test splits (Phase 3)
├── models/                   # Trained model checkpoints (Phase 4)
├── evaluation_results/       # Metrics and plots (Phase 5)
│
├── pipeline/                 # Main pipeline phases
│   ├── phase1_dicom_ingestion.py       ✅ Complete
│   ├── phase2a_landmark_detection.py   ✅ Complete (SMICNet)
│   ├── phase2b_roi_from_landmarks.py   ✅ Complete
│   ├── phase3_dataset_stratification.py ✅ Complete
│   ├── phase4_model_training.py        ✅ Complete
│   └── phase5_model_evaluation.py      ✅ Complete
│
├── models/                   # Model architecture
│   ├── resnet3d.py           # 3D ResNet-18 with CBAM
│   └── cbam.py               # CBAM attention module
│
├── data/                     # Data loading
│   ├── dataset.py            # PyTorch Dataset
│   └── transforms.py         # MONAI augmentation
│
├── evaluation/               # Evaluation utilities
│   ├── metrics.py            # AUC, bootstrap CI, thresholds
│   ├── visualization.py      # ROC curves, confusion matrices
│   └── gradcam.py            # 3D Grad-CAM
│
├── docs/                     # Documentation
│   ├── PHASE3_GUIDE.md       # Stratification guide
│   ├── PHASE4_GUIDE.md       # Training guide
│   └── PHASE5_GUIDE.md       # Evaluation guide
│
├── notebooks/                # Demo notebooks
│   └── demo_full_pipeline.ipynb
│
├── smicnet/                  # Submodule: Landmark detection
├── utils/                    # Utilities and viewers
├── labels.csv                # Surgical findings
└── requirements.txt          # Dependencies
```

---

## Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda create -n ear_ai python=3.9
conda activate ear_ai

# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA (recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### 2. Run Pipeline

```bash
# Phase 1: DICOM Processing
python pipeline/phase1_dicom_ingestion.py

# Phase 2: Landmark Detection & ROI Extraction
python pipeline/phase2a_landmark_detection.py
python pipeline/phase2b_roi_from_landmarks.py

# Phase 3: Dataset Stratification
python pipeline/phase3_dataset_stratification.py

# Phase 4: Model Training (per fold)
python pipeline/phase4_model_training.py --fold 0 --epochs 100

# Phase 5: Model Evaluation
python pipeline/phase5_model_evaluation.py --generate_gradcam
```

---

## Pipeline Phases

### ✅ Phase 1: DICOM Ingestion

Preprocesses raw DICOM files into standardized volumes.

- Loads DICOM with correct Z-ordering
- Applies correct bone windowing (Range: -1300 to 2700 HU)
- Splits left/right temporal bones
- Outputs: `processed_data/pt_XX/[left|right]/`

### ✅ Phase 2: ROI Extraction (SMICNet)

Extracts anatomically centered 128³ ROIs using deep learning landmarks.

- Phase 2A: Automatic landmark detection (Apex, Basal Turn, Round Window)
- Phase 2B: Fixed 128×128×128 ROI centered on middle ear
- Outputs: `roi_extracted/pt_XX/[left|right]/axial_roi.npy`

### ✅ Phase 3: Dataset Stratification

Patient-level stratified cross-validation with test set holdout.

- Multi-label stratification (cholesteatoma + ossicular)
- 5-fold CV with 15% test set
- Outputs: `dataset_splits/fold_*.json`, `test_set.json`

### ✅ Phase 4: Model Training

3D ResNet-18 with CBAM attention and multi-task learning.

- MedicalNet pre-trained weights
- Mixed precision training
- Cosine annealing LR schedule
- Outputs: `models/fold_*/best_checkpoint.pth`

### ✅ Phase 5: Model Evaluation

Comprehensive evaluation with confidence intervals.

- Ensemble predictions from 5-fold models
- Bootstrap 95% CI for AUC, sensitivity, specificity
- Grad-CAM interpretability visualizations
- Error analysis (FP/FN cases)
- Outputs: `evaluation_results/`

---

## Validation Pipeline

For quick pipeline validation with reduced resources:

```bash
# Use validation scripts (2-head, fewer epochs)
python pipeline/phase3_dataset_stratification_validation.py
python pipeline/phase4_model_training_validation.py --fold 0 --epochs 50
python pipeline/phase5_model_evaluation_validation.py
```

---

## Utilities

| Tool                                 | Description              |
| ------------------------------------ | ------------------------ |
| `python -m utils.viewer_interactive` | Interactive slice viewer |
| `python -m utils.validation`         | Data quality checks      |
| `python -m utils.generate_report`    | Processing statistics    |

---

## Documentation

See `docs/` for detailed guides:

- [Phase 3: Dataset Stratification](docs/PHASE3_GUIDE.md)
- [Phase 4: Model Training](docs/PHASE4_GUIDE.md)
- [Phase 5: Model Evaluation](docs/PHASE5_GUIDE.md)

Interactive notebook: `notebooks/demo_full_pipeline.ipynb`

---

**Status:** All Phases Complete ✅
