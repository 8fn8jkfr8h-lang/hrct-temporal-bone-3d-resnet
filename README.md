# Temporal Bone HRCT - AI Pathology Detection

AI-based detection of middle ear pathologies from HRCT temporal bone scans.

---

## Project Structure

```
HRCT_Temporal/
├── dicom_input/              # Raw DICOM files
├── processed_data/           # Preprocessed volumes
├── visualizations/           # Generated overview images
│
├── pipeline/                 # Main pipeline phases
│   ├── phase1_dicom_ingestion.py       ✅ Complete
│   ├── phase2_roi_extraction.py        🔄 TODO
│   ├── phase3_dataset_stratification.py 🔄 TODO
│   ├── phase4_model_training.py        🔄 TODO
│   └── phase5_model_evaluation.py      🔄 TODO
│
├── utils/                    # Utilities and tools
│   ├── dicom_processor.py   # DICOM processing
│   ├── validation.py        # Data validation
│   ├── generate_report.py   # Statistics report
│   ├── viewer_interactive.py # Interactive viewer
│   └── viewer_batch.py      # Batch overview
│
├── labels.csv               # Surgical findings
├── requirements.txt         # Dependencies
└── temporal_bone_project.md # Full specification
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Process DICOM Files

```bash
python pipeline/phase1_dicom_ingestion.py
```

### 3. Validate Results

```bash
python -m utils.validation
```

### 4. Manual Verification

```bash
# Interactive viewer
python -m utils.viewer_interactive

# Or batch overview
python -m utils.viewer_batch
```

### 5. Generate Report

```bash
python -m utils.generate_report
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
6. Saves processed volumes

**Output:** `processed_data/pt_XX/[left|right]/`
- `axial_volume.npy` - Axial slices (normalized 0-1)
- `coronal_volume.npy` - Coronal slices (normalized 0-1)
- `metadata.json` - Spacing, dimensions, patient info

**Note:** Coronal views are reconstructed from the full volume. The machine learning pipeline (Phase 2 & 3) processes data based on array indices, not visual orientation. The data is topologically correct (Index 0 = Feet, Index N = Head).

---

### 🔄 Phase 2: ROI Extraction (TODO)

Extract middle ear regions from processed volumes.

**Planned:**
1. Temporal bone localization
2. Anatomical landmark detection
3. Optimal slice range selection
4. Middle ear ROI cropping (224×224)
5. Laterality standardization

---

### 🔄 Phase 3-5: Model Development (TODO)

- Phase 3: Dataset stratification (train/val/test split)
- Phase 4: Model training (3D ResNet-50 dual-stream)
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

**Note:** The viewer applies a vertical flip to the coronal view for correct anatomical display (Head at top).

### Batch Viewer

Generate overview images:

```bash
python -m utils.viewer_batch
```

Output: `visualizations/pt_XX_overview.png`

**Layout:** 4 rows × 3 columns
- Rows 1-2: Left ear (3 axial slices + 3 coronal slices)
- Rows 3-4: Right ear (3 axial slices + 3 coronal slices)

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

## Loading Data

```python
import numpy as np
import json

# Load volumes
axial = np.load('processed_data/pt_01/left/axial_volume.npy')
coronal = np.load('processed_data/pt_01/left/coronal_volume.npy')

# Load metadata
with open('processed_data/pt_01/left/metadata.json') as f:
    meta = json.load(f)

print(f"Axial: {axial.shape}")    # (slices, 768, 404)
print(f"Coronal: {coronal.shape}")  # (slices, height, width)
print(f"Spacing: {meta['pixel_spacing']} mm")
```

---

## Technical Details

### Preprocessing Pipeline

1. **Slice Sorting:** By `ImagePositionPatient[2]` (Z-coordinate), not filename
2. **HU Conversion:** `HU = pixel_value × RescaleSlope + RescaleIntercept`
3. **Bone Windowing:** Width=4000, Level=700, normalized to [0, 1]
4. **Coronal Reconstruction:** SimpleITK multiplanar reformation (orthogonal) from full volume, isotropic 0.335mm
5. **Lateral Split:** Midline at x=384 with 20px overlap for both axial and coronal

**Design Rationale:** Coronal reconstruction is done on the full volume before splitting to maintain proper anatomical proportions. Phase 2 ROI extraction will crop to temporal bone regions in both views.

### Data Format

**Axial volumes:** `(slices, 768, 404)` float32, range [0, 1] - half-width after split  
**Coronal volumes:** `(slices, height, width)` float32, range [0, 1] - half-width after split  
**Metadata:** JSON with spacing, dimensions, patient info

### Visualization
The raw data is stored with index 0 = Inferior (Feet). The visualization tools (`viewer_interactive.py` and `viewer_batch.py`) apply a vertical flip to the coronal view so that it appears "Head-up" for human inspection. The ML pipeline consumes the raw data without this flip.

### Validation Results

✅ All processed data passed validation:
- No NaN or Inf values
- Values in [0, 1] range
- Correct dimensions
- Complete metadata
- Proper anatomical orientation

---

## Troubleshooting

**JPEG-LS decompression error:**
```bash
pip install pylibjpeg pylibjpeg-libjpeg
```

**No DICOM files found:**  
Ensure files are in `dicom_input/pt_XX/` with pattern `0_*`

**Import errors:**
```bash
pip install -r requirements.txt
```

---

## Next Steps

1. Manual verification using interactive viewer
2. Implement Phase 2: ROI extraction
3. Continue with dataset stratification and model training

---

**Status:** Phase 1 Complete ✅ | Ready for Phase 2