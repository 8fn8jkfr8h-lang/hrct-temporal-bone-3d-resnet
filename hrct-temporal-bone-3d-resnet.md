# PROJECT SUMMARY: AI-Based Temporal Bone HRCT Pathology Detection

## **TEAM COMPOSITION**

- **Radiology Resident (PGY1):** Model development, technical implementation, analysis
- **ENT Resident (PGY2):** Dataset curation, surgical findings documentation, clinical validation
- **Project Type:** Joint research thesis + peer-reviewed publication
- **Hardware:** High-end GPU available (no constraints)

---

## **CLINICAL PROBLEM**

**Objective:** Develop an AI system to detect middle ear pathologies on HRCT temporal bone scans and validate against intraoperative findings (gold standard).

**Target Pathologies (Priority Order):**

1. **Cholesteatoma** (PRIMARY - most clinically important)
2. **Ossicular chain discontinuity** (SECONDARY)
3. **Facial nerve dehiscence** (EXPLORATORY - if reliably documented)

**Clinical Context:**

- Preoperative HRCT assessment is standard of care
- Accurate detection guides surgical planning
- Current limitation: Radiologist interpretation variability
- This study validates AI against actual surgical findings (rare in literature)

---

## **DATASET CHARACTERISTICS**

### **Patient Cohort:**

- **Total:** ~100 patients with HRCT scans
- **Expected final:** ~80-90 patients after exclusions
- **All cases:** Confirmed surgical findings available
- **Distribution:** Mix of bilateral and unilateral disease
- **Negative controls:** Normal ears confirmed intraoperatively
- **Pathology breakdown:** Unknown until labeling complete
- **Validation approach:** AI predictions vs intraoperative findings ONLY

### **Inclusion Criteria:**

- ✅ First-time temporal bone surgery (primary cases only)
- ✅ No prior middle ear/mastoid surgery on that ear
- ✅ Complete HRCT thin slice series available
- ✅ Confirmed intraoperative findings documented
- ✅ Adequate image quality

### **Exclusion Criteria:**

- ❌ Revision/redo mastoidectomy or tympanoplasty
- ❌ Any prior middle ear surgery on ipsilateral ear
- ❌ Congenital temporal bone malformations
- ❌ Temporal bone trauma with fractures
- ❌ Incomplete HRCT series
- ❌ Poor image quality (severe artifacts, motion)
- ❌ For bilateral cases: exclude the revision ear, keep primary ear only

### **HRCT Imaging Specifications:**

**Scanner:** Philips Ingenuity Core 128

**Key Parameters:**

- **Slice thickness:** 0.67mm (excellent for temporal bone)
- **Slice spacing:** 0.335mm
- **Pixel spacing:** 0.229mm × 0.229mm
- **Matrix:** 768 × 768 pixels
- **Orientation:** Axial (ImageOrientationPatient: [1,0,0,0,1,0])
- **Reconstruction kernel:** YE (bone kernel)
- **KVP:** 120
- **Acquisition:** Spiral CT

**Data Organization:**

```
pt_01/
  ├─ 0_0.dcm
  ├─ 0_10.dcm
  ├─ 0_20.dcm
  ...
  └─ 0_N.dcm  (variable slices per patient, typically 150-200)

pt_02/
  ├─ 0_0.dcm
  ...
```

**File Naming Convention:** Files named `0_0`, `0_10`, `0_20`... `0_N` (note: extra zero in hundreds place)
**Slices per patient:** Variable (typically 150-200), all slices present
**File size:** 100-150 MB per patient
**Series type:** HRCT thin slices only (excluding soft tissue reconstructions)

**Important:** Coronal images must be reconstructed from axial (no native coronal available)

---

## **CORE TECHNICAL CHALLENGES**

1. **Variable slice counts:** Each patient has different number of slices (150-200 range)
2. **Variable ear positions:** Left and right temporal bones at different slice levels per patient
3. **Non-standardized acquisition:** Different positions/orientations across patients
4. **Slice selection:** Need to extract ~20-40 relevant slices from 150-200 total per ear
5. **Class imbalance:** Unknown distribution, likely more cholesteatoma cases (surgical cohort bias)
6. **Small dataset:** ~80-90 patients = ~120-150 ears after exclusions
7. **Facial nerve documentation:** May be inconsistently reported in surgical notes
8. **Revision anatomy exclusion:** Need to carefully screen and exclude altered anatomy

---

## **PROJECT ARCHITECTURE**

### **Phase 1: Data Infrastructure & Lateral Splitting**

**DICOM Processing:**

- Load DICOM series with proper slice ordering (sort by ImagePositionPatient[2])
- Verify slice continuity and spacing consistency
- Apply bone windowing (W:4000, L:700) -> Clip range [-1300, 2700]
- Extract pixel arrays and metadata
- Quality checks for completeness and artifacts

**Coronal Reconstruction:**

- Use SimpleITK resampling for multiplanar reformats
- Maintain isotropic spacing (0.335mm based on acquisition)
- Generate coronal views from FULL volume first
- Output both axial and coronal volumes

**Lateral Split (Left/Right Separation):**

- Performed on BOTH axial and coronal volumes after reconstruction
- Sagittal plane splitting at midline (x = width/2)
- 20px margin overlap at midline to prevent edge artifacts
- DICOM convention: Right side of image = Left side of patient
- Each ear processed and saved independently from this point onwards

**Why split in Phase 1:**

- Ensures consistent separation across all subsequent processing steps
- Reduces memory footprint for Phase 2+ operations (half-volume processing)
- Aligns with clinical workflow (each ear is independent diagnostic unit)
- Simplifies ROI extraction in Phase 2 (no lateral disambiguation needed)

**Output Structure:**

```
processed_data/
├─ pt_01/
│  ├─ left/
│  │  ├─ axial_volume.npy       # (~N, 768, ~404) - half width + margin
│  │  ├─ coronal_volume.npy     # Reconstructed, half width
│  │  └─ metadata.json          # spacing, side='L', dimensions
│  │
│  └─ right/
│     ├─ axial_volume.npy       # (~N, 768, ~404) - half width + margin
│     ├─ coronal_volume.npy     # Reconstructed, half width
│     └─ metadata.json          # spacing, side='R', dimensions
│
├─ pt_02/
│  ├─ left/
│  │  └─ ... (same structure)
│  └─ right/
│     └─ ... (same structure)
...
```

---

### **Phase 2: ROI Extraction Pipeline (SMICNet Integration)**

**Selected Approach: Deep Learning Landmark Detection (SMICNet)**

We utilize the pre-trained **SMICNet** model (specialized for cochlear landmark detection in CT) to robustly identify the cochlea, which serves as a stable anatomical anchor for the middle ear.

#### **Stage 1: Automatic Landmark Detection**

**Script:** `pipeline/phase2a_landmark_detection.py`

**Input:** Pre-split hemicranial volume from Phase 1 (`axial_volume.npy`).

**Method:**

1. **Coarse Search:** Scan the volume with a sliding window (large stride) to identify the general region of the cochlea using the SMICNet classifier.
2. **Fine Search:** Perform a dense sliding window search around the best candidate region to generate probability maps for three landmarks:
   - Apex of Cochlea
   - Basal Turn of Cochlea
   - Round Window
3. **Centroid Extraction:** Compute the weighted center of mass for each probability map to obtain sub-voxel coordinates (x, y, z).

**Output:** JSON file containing the 3D coordinates of the detected landmarks for each ear.

#### **Stage 2: Geometric ROI Extraction**

**Script:** `pipeline/phase2b_roi_from_landmarks.py`

**Input:** Detected landmarks and the original volume.

**Method:**

1. **Center Calculation:** Derive the center of the middle ear ROI using anatomical offsets relative to the detected **Basal Turn**.
   - **Z (Vertical):** Aligned with the Basal Turn.
   - **Y (Anterior-Posterior):** Centered between the Apex and Basal Turn.
   - **X (Lateral):** Offset laterally (outwards) from the Basal Turn to capture the middle ear cavity.
2. **Fixed Z-Bounding (Center & Expand):**
   - **Target Z-depth:** Fixed 128 slices (~42.88mm) to ensure coverage of Tegmen and Jugular Bulb.
   - **Algorithm:** $z_{start} = c_z - 64$, $z_{end} = c_z + 64$.
   - **Padding:** If the calculated range exceeds volume limits (top or bottom), pad with "Air" (value 0).
3. **Extraction:** Crop the fixed volume: `(128, 128, 128)`.

**Output:**

- `axial_roi.npy`: The cropped 3D volume containing the middle ear (128, 128, 128).
- `roi_metadata.json`: Metadata including bounds and QC metrics.
- `roi_preview.png`: Visualization of the center slice.

**Advantages:**

- **Robustness:** Handles anatomical variations and head tilt better than rigid registration.
- **Precision:** Anchors the ROI to the cochlea, which is a stable bony structure.
- **Completeness:** Fixed 64-slice depth ensures critical superior/inferior structures are not clipped, with attention mechanisms handling any empty padding.

**Complete Data Structure (After Phase 1 + Phase 2):**

```
processed_data/
├─ pt_01/
│  ├─ left/
│  │  ├─ axial_volume.npy
│  │  └─ ...
│  └─ right/
│     └─ ...
│
roi_extracted/
├─ pt_01/
│  ├─ left/
│  │  ├─ axial_roi.npy             # (64, 224, 224) - Final ROI
│  │  ├─ roi_metadata.json         # Bounds, center, QC info
│  │  └─ roi_preview.png           # QC Image
│  └─ right/
│     └─ ...
```

---

### **Phase 3: Dataset Creation & Labeling**

**Surgical Findings Documentation:**

ENT resident creates master label spreadsheet from surgical notes.

**Label Sheet Structure:**

```csv
patient_id,ear,cholesteatoma,ossicular_discontinuity,facial_dehiscence,surgery_type,exclusion_status,notes
pt_01,L,1,1,1,primary,include,"Attic cholesteatoma, incus erosion, FN dehiscence tympanic"
pt_01,R,0,0,0,none,include,"Normal - verified intraop"
pt_02,L,1,0,NULL,primary,include,"Cholesteatoma, intact chain, FN not visualized"
pt_02,R,1,1,0,revision,EXCLUDE,"Revision mastoidectomy 2022"
```

**Labeling Protocol:**

**Cholesteatoma:**

- 0 = Absent (confirmed normal intraop OR clear on CT in non-surgical ear)
- 1 = Present (any location: attic, sinus, pars tensa, mastoid)

**Ossicular Discontinuity:**

- 0 = Intact chain (malleus-incus-stapes connected and mobile)
- 1 = Discontinuity (erosion, dislocation, fixation confirmed intraop)

**Facial Nerve Dehiscence:**

- 0 = Intact bony canal (confirmed during surgery)
- 1 = Dehiscence present (bony defect identified)
- NULL = Not documented/not visualized during surgery

**Surgery Type:**

- primary = First-time surgery for this ear
- revision = Prior surgery on same ear
- none = No surgery (contralateral normal ear)

**Exclusion Status:**

- include = Use in dataset
- exclude = Remove from dataset (revision cases, poor quality, etc.)

**Critical Rules:**

- Label each ear independently (bilateral cases = separate rows)
- Only include patients with surgical confirmation
- For bilateral cases: exclude any ear with prior surgery
- Document NULL for facial nerve if not reliably assessed

**Time Estimate:** 10-18 hours for ENT resident to complete all labeling

---

### **Phase 4: Dataset Stratification & Splitting**

**After Exclusions and Labeling:**

```
Expected dataset composition:
├─ Total patients: ~80-90 (after excluding revisions)
├─ Total ears: ~120-150 (some bilateral primaries)
├─ Cholesteatoma distribution: Unknown (will assess)
├─ Ossicular distribution: Unknown (will assess)
└─ Facial dehiscence: Likely incomplete documentation
```

**Stratified Split Strategy:**

```
Patient-level splitting (prevents data leakage):
├─ Training: 70% of patients (~56-63 patients)
├─ Validation: 15% of patients (~12-14 patients)
└─ Test: 15% of patients (~12-14 patients)

Stratification:
├─ Primary stratification: Cholesteatoma presence (main outcome)
├─ Secondary consideration: Ossicular status
└─ Ensure bilateral cases stay together (all ears from one patient in same split)
```

**Class Imbalance Assessment:**

```
After splitting, analyze:
├─ Positive/negative ratio for each pathology
├─ Class weights for loss function
├─ Need for oversampling/undersampling
└─ Optimal decision thresholds
```

---

### **Phase 5: Model Development**

**Architecture: MedicalNet (3D ResNet-18) with CBAM Attention**

We leverage **MedicalNet**, a 3D ResNet pre-trained on a large aggregation of medical datasets (lungs, brains, livers), ensuring the model understands 3D anatomical features better than random initialization or Kinetics-700 (video) pre-training.

#### **Core Architecture**

**1. The Backbone: ResNet-18 (3D)**

- **Choice:** ResNet-18 is selected as the "Goldilocks" model.
  - ResNet-10 is too shallow for complex bone erosion features.
  - ResNet-50 is too large for our dataset size (~100 patients), risking overfitting.
- **Weights:** Initialize with `resnet_18_23dataset.pth` (MedicalNet weights).

**2. Input Modification: Single Channel**

- MedicalNet expects multi-channel input by default.
- **Modification:** Replace the first convolutional layer to accept **1 channel** (Grayscale CT).
  ```python
  model.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
  ```

**3. Attention Mechanism: CBAM (Convolutional Block Attention Module)**

- **Purpose:** To handle the fixed 64-slice input which may contain "air" padding. The attention module learns to "downweight" empty padding and focus on the bony anatomy.
- **Placement:** Inserted after each Residual Block (Layer 1, Layer 2, Layer 3).
- **Components:**
  - **Channel Attention:** Focuses on _what_ features are meaningful.
  - **Spatial Attention:** Focuses on _where_ the informative regions are (ignoring air).

#### **Classification Heads**

(Separate heads for each pathology attached to the global pooling output of the backbone)

1.  **Cholesteatoma Head:** FC -> ReLU -> Dropout -> FC -> Sigmoid
2.  **Ossicular Head:** FC -> ReLU -> Dropout -> FC -> Sigmoid
3.  **Facial Nerve Head:** FC -> ReLU -> Dropout -> FC -> Sigmoid

---

#### **Training Strategy**

**Loss Function:**

```python
Multi-Task Weighted Loss:

L_total = w1 × L_cholesteatoma + w2 × L_ossicular + w3 × L_facial

Where:
├─ L_cholesteatoma: Focal Loss or Weighted BCE (handle imbalance)
├─ L_ossicular: Focal Loss or Weighted BCE
├─ L_facial: Weighted BCE with sample weighting (account for NULL labels)
└─ Weights: w1=0.5, w2=0.3, w3=0.2 (prioritize cholesteatoma)

Focal Loss formulation (if severe class imbalance):
FL(p) = -α(1-p)^γ log(p)
├─ α: class balance weight
├─ γ: focusing parameter (typically 2)
└─ Focuses learning on hard misclassified examples
```

**Handling NULL Labels (Facial Nerve):**

```python
For cases where facial_dehiscence = NULL:
├─ Exclude from loss computation for facial nerve head
├─ Use sample_weight = 0 for these cases
├─ Train only on cases with confirmed documentation
└─ Report performance only on documented subset
```

**Optimizer & Learning Rate Schedule:**

```python
Optimizer: AdamW
├─ Initial learning rate: 1e-4 for pretrained backbone
├─ Initial learning rate: 1e-3 for classification heads
├─ Weight decay: 1e-4
└─ Gradient clipping: max_norm = 1.0

Learning Rate Schedule: Cosine Annealing with Warm Restarts
├─ T_0: 10 epochs (initial restart period)
├─ T_mult: 2 (double restart period each cycle)
├─ η_min: 1e-6 (minimum learning rate)
└─ Allows escape from local minima

Alternative: OneCycleLR
├─ Max LR: 1e-3
├─ Total steps: determined by epochs
└─ Generally faster convergence
```

**Regularization:**

```python
Dropout: 0.3-0.4 in classification heads

Label Smoothing: 0.1
├─ Prevents overconfident predictions
├─ Improves calibration
└─ Particularly useful for small datasets

MixUp Augmentation (optional):
├─ Mix pairs of training samples
├─ Creates synthetic interpolated examples
└─ Proven effective in medical imaging

Early Stopping:
├─ Monitor: Validation AUC (average across pathologies)
├─ Patience: 25 epochs
└─ Restore best weights
```

**Data Augmentation:**

```python
Training Augmentation (3D):
├─ Random 3D rotation: ±15° (anatomically valid)
├─ Random 3D translation: ±10 pixels
├─ Random scaling: 0.9-1.1
├─ Elastic deformation: subtle (preserve anatomy)
├─ Intensity augmentation:
│   ├─ Random brightness: ±15%
│   ├─ Random contrast: ±15%
│   ├─ Random gamma: 0.8-1.2
│   └─ Gaussian noise: σ=0.02
├─ Random Gaussian blur: σ=0.5-1.5
└─ No horizontal flipping (already standardized)

Validation/Test Augmentation:
└─ None (only normalization)

Test-Time Augmentation (TTA):
├─ Apply augmentations at inference
├─ Average predictions across augmented versions
├─ Typically improves performance 1-2% AUC
└─ Use for final test set evaluation
```

**Training Hyperparameters:**

```
Batch size: 16-32 (with high-end GPU)
├─ Larger batches improve batch norm statistics
└─ Use gradient accumulation if memory constrained

Epochs: 150-200 (with early stopping)
├─ Small dataset needs more epochs
└─ Monitor for overfitting

Mixed Precision Training: FP16
├─ Faster training
├─ Reduced memory usage
└─ Minimal accuracy impact with modern GPUs

Gradient Accumulation: 2-4 steps if needed
└─ Effective batch size = batch_size × accumulation_steps
```

---

#### **Handling Class Imbalance**

**Assessment After Labeling:**

```
For each pathology, calculate:
├─ Positive/negative ratio
├─ Minority class sample count
└─ Imbalance severity
```

**Strategy Selection Based on Imbalance:**

**Mild Imbalance (ratio 1:2 to 1:3):**

```
├─ Weighted Binary Cross-Entropy
└─ pos_weight = (num_negative / num_positive)
```

**Moderate Imbalance (ratio 1:3 to 1:5):**

```
├─ Focal Loss with α=0.25, γ=2
└─ Oversample minority class by 1.5x
```

**Severe Imbalance (ratio > 1:5):**

```
├─ Focal Loss with α=0.25, γ=2
├─ Oversample minority class by 2x
├─ Undersample majority class by 0.7x
└─ Adjust decision threshold on validation set
```

**Threshold Optimization:**

```
For each pathology:
├─ Generate predictions on validation set
├─ Sweep thresholds from 0.1 to 0.9
├─ Calculate F1-score at each threshold
├─ Select threshold maximizing F1 (or optimizing for sensitivity if clinically preferred)
└─ Apply optimized threshold on test set
```

---

### **Phase 6: Advanced Techniques (If Time/Resources Permit)** - OPTIONAL

**Ensemble Methods:**

```
Train 5 models with different:
├─ Random initialization seeds
├─ Train/val splits (5-fold cross-validation)
└─ Augmentation strategies

Ensemble prediction:
├─ Average probabilities across 5 models
├─ Typically improves AUC by 2-4%
└─ More robust to individual model failures
```

**Semi-Supervised Learning (If unlabeled data available):**

```
If ENT has additional HRCT scans without surgical confirmation:
├─ Use pseudo-labeling on unlabeled data
├─ Self-training: Model generates labels for unlabeled data
├─ Retrain on combined labeled + pseudo-labeled data
└─ Can improve performance with limited labeled data
```

**Transfer Learning from Related Tasks:**

```
Consider pretraining on:
├─ MedicalNet (3D medical imaging pretrained weights)
├─ Kinetics-700 (action recognition, 3D CNN pretrained)
├─ Or other temporal bone datasets if publicly available
└─ Fine-tune on your dataset
```

---

### **Phase 7: Validation & Performance Analysis**

#### **Primary Performance Metrics**

**For Each Pathology (Cholesteatoma, Ossicular, Facial Nerve):**

```
Core Metrics:
├─ Sensitivity (Recall): TP / (TP + FN)
├─ Specificity: TN / (TN + FP)
├─ Positive Predictive Value (PPV/Precision): TP / (TP + FP)
├─ Negative Predictive Value (NPV): TN / (TN + FN)
├─ F1-Score: 2 × (Precision × Recall) / (Precision + Recall)
├─ Accuracy: (TP + TN) / (TP + TN + FP + FN)
└─ AUC-ROC: Area under receiver operating characteristic curve

With 95% Confidence Intervals:
└─ Bootstrap resampling (1000 iterations) for all metrics
```

**Threshold Selection:**

```
Cholesteatoma (Primary Outcome):
└─ Optimize for F1-score (balance sensitivity/PPV)

Ossicular Discontinuity:
└─ May prefer higher sensitivity (intraop awareness critical)

Facial Nerve Dehiscence (Exploratory):
└─ Optimize for specificity (avoid false alarms)
```

**Multi-Threshold Analysis:**

```
Report performance at multiple operating points:
├─ High sensitivity threshold (e.g., 90% sensitivity)
├─ Balanced threshold (maximize F1)
├─ High specificity threshold (e.g., 90% specificity)
└─ Allows clinicians to choose operating point
```

---

#### **Error Analysis Framework**

**Confusion Matrix Deep Dive:**

```
For each misclassified case:

1. CT Image Review:
   ├─ Re-examine original axial and coronal slices
   ├─ Check for subtle findings missed by model
   ├─ Assess image quality (artifacts, noise, resolution)
   └─ Compare to correctly classified similar cases

2. Grad-CAM/Attention Visualization:
   ├─ Generate heatmaps showing where model focused
   ├─ Verify attention on anatomically relevant regions
   ├─ Identify if model using spurious correlations
   └─ Check attention distribution across slices

3. Surgical Notes Review:
   ├─ Verify ground truth labeling accuracy
   ├─ Check for ambiguous documentation
   ├─ Assess if intraop findings were definitive
   └─ Consider CT-surgery discordance

4. Error Categorization:

   False Positives (Model says YES, Surgery says NO):
   ├─ Inflammation/granulation mimicking cholesteatoma
   ├─ Fluid/mucosal thickening misclassified
   ├─ Normal anatomical variants (high jugular bulb, etc.)
   ├─ Image artifacts causing false appearance
   └─ Model overconfident on equivocal findings

   False Negatives (Model says NO, Surgery says YES):
   ├─ Small/subtle cholesteatoma (< 3mm)
   ├─ Unusual location (sinus tympani, facial recess)
   ├─ Poor CT quality in relevant region
   ├─ ROI extraction missed critical slices
   ├─ Partial volume averaging
   └─ Early/minimal disease

5. Clinical Significance Assessment:
   ├─ Would error impact patient management?
   ├─ Was finding equivocal even to radiologist?
   ├─ Is error pattern systematic or random?
   └─ Can error be mitigated with clinical context?

6. Documentation:
   └─ Create detailed case report for each misclassification
       (include images, heatmaps, surgical notes, analysis)
```

**Subgroup Analysis:**

```
Stratify performance by:

Disease Characteristics:
├─ Cholesteatoma size: small (<5mm) vs medium (5-10mm) vs large (>10mm)
├─ Cholesteatoma location: attic vs sinus vs pars tensa vs mastoid
├─ Ossicular erosion: partial vs complete, which ossicle affected
├─ Facial dehiscence location: labyrinthine vs tympanic vs mastoid segment
└─ Bilateral vs unilateral disease

Image Quality:
├─ High quality (no artifacts, full coverage)
├─ Moderate quality (minor artifacts)
├─ Low quality (significant artifacts, limited coverage)
└─ Slice thickness variation if any

Patient Demographics (if available):
├─ Age groups
├─ Gender
└─ Duration of symptoms

Surgical Complexity:
├─ Extensive disease vs limited
├─ Canal wall up vs canal wall down surgery
└─ Ossiculoplasty performed vs not
```

**Inter-Observer Agreement (If Resources Available):**

```
Optional but valuable:
├─ Have 2-3 radiologists independently read test set CTs
├─ Calculate Cohen's kappa between:
│   ├─ AI vs each radiologist
│   ├─ Radiologists vs each other
│   └─ Each reader vs surgical gold standard
└─ Demonstrates AI performance in context of human variability
```

---

#### **Interpretability: Grad-CAM Analysis**

**Implementation:**

```python
Grad-CAM for 3D Volumetric Data:

Target Layer: Last 3D convolutional layer in each stream

For each test case:
├─ Generate 3D Grad-CAM heatmap for cholesteatoma prediction
├─ Generate 3D Grad-CAM heatmap for ossicular prediction
├─ Generate 3D Grad-CAM heatmap for facial nerve prediction
└─ Process both axial and coronal stream activations

Visualization:
├─ Project 3D heatmap onto 2D slices (MIP or slice-wise)
├─ Overlay on original CT (use jet or hot colormap)
├─ Create side-by-side comparison: CT | Heatmap | Overlay
└─ Generate for central slices showing pathology
```

**Qualitative Assessment Categories:**

```
1. Correct Localization (True Positives):
   ├─ Model highlights attic/antrum in cholesteatoma cases
   ├─ Model focuses on ossicular chain in discontinuity cases
   ├─ Model attends to facial nerve canal in dehiscence cases
   └─ Demonstrates anatomically appropriate reasoning

2. Appropriate Attention (True Negatives):
   ├─ Model shows diffuse/non-specific attention in normal cases
   ├─ No strong focal activation on any pathological region
   └─ Confirms model not using spurious features

3. Explainable Errors (False Positives):
   ├─ Model highlights inflammatory changes mimicking cholesteatoma
   ├─ Attention on fluid/granulation tissue
   └─ Shows model reasoning, even if incorrect

4. Missed Findings (False Negatives):
   ├─ Subtle pathology in region of low attention
   ├─ Model focused on wrong anatomical region
   └─ Reveals model limitations

5. Spurious Correlations Check:
   ├─ Verify model not using non-anatomical features
   ├─ Check attention not on scanner artifacts
   ├─ Ensure attention appropriate across all slices
   └─ Validate clinical plausibility
```

**Publication Figures:**

```
Figure 3: Grad-CAM Interpretability Examples

Layout (4×4 grid):

Row 1 - Cholesteatoma True Positives:
├─ Case A: Attic cholesteatoma with correct localization
├─ Case B: Mastoid cholesteatoma with correct localization
├─ Case C: Pars tensa cholesteatoma with correct localization
└─ Case D: Bilateral cholesteatoma with bilateral highlights

Row 2 - True Negatives:
├─ Normal case A: Diffuse attention, no focal pathology
├─ Normal case B: Appropriate anatomical focus
├─ Post-inflammatory case: Model correctly ignores scarring
└─ Fluid-filled case: Model correctly distinguishes from cholesteatoma

Row 3 - Ossicular & Facial Nerve Examples:
├─ Ossicular discontinuity: Attention on gap in chain
├─ Ossicular normal: Diffuse attention on intact chain
├─ Facial dehiscence: Attention on canal defect (coronal view)
└─ Facial intact: Diffuse attention on normal canal

Row 4 - Error Cases with Explanations:
├─ False positive: Inflammation highlighted, resembles cholesteatoma
├─ False negative: Subtle small cholesteatoma, insufficient attention
├─ Difficult case: Equivocal even for radiologist
└─ Artifact case: Model appropriately ignores motion artifact

Each panel:
├─ Original CT slice
├─ Grad-CAM heatmap overlay
├─ Prediction probability
├─ Ground truth label
└─ Brief caption explaining finding
```

---

### **Phase 8: Statistical Analysis**

**Primary Statistical Tests:**

```
Hypothesis Testing:

H0: AI system has sensitivity ≤ 75% for cholesteatoma detection
H1: AI system has sensitivity > 75% for cholesteatoma detection
├─ One-sample proportion test
└─ Power calculation (post-hoc)

Secondary Hypotheses:
├─ AI specificity ≥ 80% for cholesteatoma
├─ AI AUC ≥ 0.85 for cholesteatoma
├─ AI sensitivity ≥ 70
```
