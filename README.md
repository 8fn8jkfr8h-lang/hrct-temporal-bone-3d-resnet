# Temporal Bone HRCT - AI Pathology Detection

AI-based detection of middle ear pathologies from HRCT temporal bone scans.

---

## Project Structure

```
HRCT_Temporal/
├── dicom_input/              # Raw DICOM files
├── processed_data/           # Preprocessed volumes (Phase 1)
├── landmarks_detected/       # Detected 3D landmarks (Phase 2A)
├── roi_extracted/            # Final Middle Ear ROIs (Phase 2B)
├── visualizations/           # Generated overview images
│
├── pipeline/                 # Main pipeline phases
│   ├── phase1_dicom_ingestion.py       ✅ Complete
│   ├── phase2a_landmark_detection.py   ✅ Complete (SMICNet)
│   ├── phase2b_roi_from_landmarks.py   ✅ Complete
│   ├── phase3_dataset_stratification.py 🔄 TODO
│   ├── phase4_model_training.py        🔄 TODO
│   └── phase5_model_evaluation.py      🔄 TODO
│
├── smicnet/                  # Submodule: Landmark detection
│   ├── TrainedNetworkWeights/  # Pre-trained models
│   └── ...
│
├── utils/                    # Utilities and tools
│   ├── dicom_processor.py    # DICOM processing
│   ├── validation.py         # Data validation
│   ├── generate_report.py    # Statistics report
│   ├── viewer_interactive.py # Interactive viewer
│   ├── viewer_batch.py       # Batch overview
│   └── roi_reviewer.py       # ROI review tool (Phase 2C)
│
├── labels.csv                # Surgical findings
├── requirements.txt          # Dependencies
└── temporal_bone_project.md  # Full specification
```

---

## Quick Start

### 1. Install Dependencies

```bash
# Recommended: Use Conda for GPU support
conda create -n ear_ai python=3.9
conda activate ear_ai
pip install -r requirements.txt
```

### 2. Process DICOM Files

```bash
python pipeline/phase1_dicom_ingestion.py
```

### 3. Detect Landmarks (SMICNet)

```bash
python pipeline/phase2a_landmark_detection.py
```

### 4. Extract ROIs

```bash
python pipeline/phase2b_roi_from_landmarks.py
```

### 5. Validate Results

```bash
python -m utils.validation
```

---

## Pipeline Phases

### ✅ Phase 1: DICOM Ingestion (Complete)

Preprocesses raw DICOM files into standardized volumes.

**What it does:**
1. Loads DICOM files sorted by Z-position (Inferior -> Superior)
2. Converts to Hounsfield Units
3. Applies bone windowing (Width: 4000, Level: 700)
4. Reconstructs coronal views from full volume (isotropic 0.335mm)
5. Splits left/right temporal bones (20px overlap)
6. Filters based on `labels.csv` (`exclusion_status` column)
7. Saves processed volumes for included ears only

**Output:** `processed_data/pt_XX/[left|right]/`
- `axial_volume.npy` - Axial slices (normalized 0-1)
- `coronal_volume.npy` - Coronal slices (normalized 0-1)
- `metadata.json` - Spacing, dimensions, patient info

---

### ✅ Phase 2: SMICNet-Based ROI Extraction (Complete)

Extracts anatomically centered, bounded 3D ROIs of the middle ear using Deep Learning based landmark detection.

**What it does:**

#### Phase 2A: Automatic Landmark Detection
1. Loads pre-trained SMICNet model.
2. Performs a coarse grid search on the volume to find the general cochlea region.
3. Performs a fine sliding window search to generate probability maps for:
   - Apex
   - Basal Turn
   - Round Window
4. Extracts 3D centroids for each landmark.

#### Phase 2B: ROI Extraction
1. Calculates the middle ear center based on the detected Basal Turn and Apex.
2. Computes **fixed Z-bounds** (Center ± 64 slices).
3. Extracts a fixed 128x128 XY region and fixed 128-slice Z region (padding with air if needed).
4. Performs quality control checks (bone content, dimensions).

**Output:** `roi_extracted/pt_XX/[left|right]/`
- `axial_roi.npy` - ROI volume (128, 128, 128)
- `roi_metadata.json` - Center (x,y,z), bounds, QC results
- `roi_preview.png` - Visualization of the ROI center slice

**ROI Specifications:**
- X-Y dimensions: Fixed 128×128 pixels (42.88 mm)
- Z dimension: Fixed 128 slices (42.88 mm) with padding
- Spacing: Isotropic 0.335 mm

---

### 🔄 Phase 3: Dataset Stratification (TODO)

Train/val/test split with stratification by pathology labels.

---

### 🔄 Phase 4-5: Model Development (TODO)

- Phase 4: Model training (MedicalNet 3D ResNet-18 with CBAM Attention)
- Phase 5: Model evaluation (metrics, Grad-CAM, error analysis)

---

## Utilities

### Interactive Viewer

Manual slice-by-slice inspection:

```bash
python -m utils.viewer_interactive
```

**Controls:**
- `←/→` : Navigate slices
- `↑/↓` : Jump 10 slices
- `a/c` : Axial / Coronal view
- `l/r` : Left / Right ear
- `n/p` : Next / Previous patient
- `q` : Quit

### Validation

Check data quality:

```bash
python -m utils.validation
```

Checks: NaN/Inf values, value ranges, dimensions, metadata

### Report Generator

Statistics and pathology distribution:

```bash
python -m utils.generate_report
```

Output: `processing_report.csv`

---

**Status:** Phase 1 & 2 Complete ✅ | Ready for Phase 3
