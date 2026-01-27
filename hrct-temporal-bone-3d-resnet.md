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

### **Phase 1: Data Infrastructure**

**DICOM Processing:**
- Load DICOM series with proper slice ordering (sort by ImagePositionPatient[2])
- Verify slice continuity and spacing consistency
- Apply bone windowing (Width: 4000, Level: 700)
- Extract pixel arrays and metadata
- Quality checks for completeness and artifacts

**Coronal Reconstruction:**
- Use SimpleITK resampling for multiplanar reformats
- Maintain isotropic spacing (0.335mm based on acquisition)
- Generate coronal views through temporal bones
- Output both axial and coronal volumes

**Output Structure:**
```
processed_data/
├─ pt_01/
│  ├─ axial_volume.npy          # (N, 768, 768) where N varies
│  ├─ coronal_volume.npy        # Reconstructed
│  └─ metadata.json             # spacing, orientation, patient info
├─ pt_02/
...
```

---

### **Phase 2: ROI Extraction Pipeline**

**Selected Approach: Split-Then-Extract (Process Each Ear Independently)**

This approach splits left and right temporal bones first, then processes each side independently. This is superior because each ear can have different optimal slice ranges.

#### **Stage 1: Lateral Split (Left/Right Separation)**
```
Objective: Separate left and right temporal bone regions

Method: Sagittal plane splitting
├─ Load full CT volume (N, 768, 768)
├─ Identify midline (sagittal center plane)
│   └─ Use intensity-based detection or anatomical landmarks (nasal septum)
├─ Split into two volumes:
│   ├─ Left volume: pixels with x < midline_x (plus 20px margin)
│   └─ Right volume: pixels with x > midline_x (plus 20px margin)
├─ Crop to generous bounding boxes around each temporal bone
│   ├─ Left: x=0 to midline+20px, y=full, z=full
│   └─ Right: x=midline-20px to max, y=full, z=full
└─ Result: Two independent hemicranial volumes

Output per patient:
├─ left_volume.npy: (N, ~400, 768) - left half of head
└─ right_volume.npy: (N, ~400, 768) - right half of head
```

**Why generous bounding box with overlap:**
- Ensures no cropping of temporal bone structures
- 20px overlap at midline prevents edge artifacts
- Disk space is not a constraint

#### **Stage 2: Independent Temporal Bone Localization (Per Side)**
```
For EACH side (left and right) independently:

Objective: Locate temporal bone within hemicranial volume

Method: Coarse localization
├─ Detect bone regions (HU > 300)
├─ Identify petrous temporal bone (densest lateral structure)
├─ Find approximate center of temporal bone mass
├─ Extract sub-volume with generous 3D bounding box
│   └─ Typical: 150×150×100 voxels
└─ This becomes the "temporal bone volume" for next stage

Output per side:
└─ temporal_bone_volume.npy: (~100, 150, 150)
```

#### **Stage 3: Anatomical Landmark Detection (Per Side)**
```
For EACH temporal bone volume independently:

Objective: Identify key landmarks for middle ear localization

Method: CNN-based landmark detector OR classical CV
├─ Detect landmarks:
│   ├─ Internal Auditory Canal (IAC)
│   ├─ Cochlea (spiral structure)
│   ├─ Vestibule
│   ├─ Horizontal Semicircular Canal (HSC)
│   ├─ External Auditory Canal (EAC)
│   └─ Middle ear cavity (air-filled space)
├─ Output 3D coordinates for each landmark
├─ Calculate middle ear centroid from landmark positions
└─ Cross-validate using anatomical relationships

This runs completely independently for left vs right:
- No coordination needed between sides
- Each ear processed with its own optimal parameters
```

**Why this approach:**
- More robust than template matching
- Leverages deep learning for complex anatomy
- Handles anatomical variations per ear
- GPU availability allows sophisticated methods
- Independent processing = simpler debugging

#### **Stage 4: Optimal Slice Range Selection (Per Side)**
```
For EACH side independently:

Objective: Extract slices containing complete middle ear anatomy

Method: Landmark-guided adaptive extraction
├─ Identify central slice (maximum middle ear cavity visibility)
├─ Use HSC position as superior reference point
├─ Use jugular bulb/carotid as inferior reference point
├─ Calculate slice range: typically 30-40 slices per ear
│   └─ ADAPTIVE: Left ear may use slices 45-80
│                Right ear may use slices 52-87
│                DIFFERENT ranges are expected and optimal!
└─ Extract this specific range for this ear

Key advantage: Each ear gets its own optimal slice range
- Accounts for anatomical asymmetry
- Maximizes pathology capture per ear
- More accurate than forcing same range for both ears

Output per side:
└─ Variable-length slice stack (30-40 slices, differs per ear)
```

#### **Stage 5: Middle Ear ROI Cropping (Per Side)**
```
For EACH slice stack independently:

Objective: Crop to middle ear region at native resolution

Method: Centered crop
├─ Use middle ear centroid from landmark detection
├─ Crop 224×224 pixels around centroid
├─ Apply to all slices in the stack
└─ Maintain native resolution (0.229mm pixel spacing)

Output per side:
└─ roi_axial.npy: (30-40, 224, 224) - variable length
```

#### **Stage 6: Coronal Reconstruction (Per Side)**
```
For EACH temporal bone volume independently:

Objective: Create coronal views for facial nerve assessment

Method: Multiplanar reformation
├─ Use SimpleITK to reconstruct coronal plane
├─ Align perpendicular to lateral semicircular canal
├─ Resample to isotropic spacing (0.335mm)
├─ Extract 20-30 coronal slices through temporal bone
├─ Crop to 224×224 around middle ear centroid
└─ Result: Coronal ROI stack

Output per side:
└─ roi_coronal.npy: (20-30, 224, 224) - variable length
```

#### **Stage 7: Laterality Standardization (Per Side)**
```
For EACH side independently:

Right ears only:
├─ Apply horizontal flip (mirror image)
├─ Now all ears appear in "left ear" canonical orientation
├─ Metadata preserves original laterality
└─ Enables single unified model training

Left ears:
└─ No transformation needed (already canonical orientation)

Output per case:
├─ Standardized ROI (all appear as "left ear")
└─ metadata.json: {"original_side": "L" or "R"}
```

**Complete Data Structure:**
```
processed_data/
├─ pt_01/
│  ├─ left/
│  │  ├─ temporal_bone_volume.npy    # (100, 150, 150)
│  │  ├─ roi_axial.npy               # (35, 224, 224) - variable
│  │  ├─ roi_coronal.npy             # (25, 224, 224) - variable
│  │  ├─ landmarks.json              # IAC, HSC, cochlea coords
│  │  └─ metadata.json               # spacing, slice range, side
│  │
│  └─ right/
│     ├─ temporal_bone_volume.npy    # (100, 150, 150)
│     ├─ roi_axial.npy               # (38, 224, 224) - DIFFERENT
│     ├─ roi_coronal.npy             # (27, 224, 224) - DIFFERENT
│     ├─ landmarks.json
│     └─ metadata.json
│
├─ pt_02/
│  ├─ left/
│  │  └─ ... (same structure)
│  └─ right/
│     └─ ... (may be excluded if revision case)
```

**Expected Output:** 
- ~120-150 independent ear ROI volumes (after exclusions)
- Each ear with its own optimal slice range (30-40 axial, 20-30 coronal)
- Each with axial and coronal views
- Consistent orientation (all standardized to "left ear" appearance)
- Variable slice counts handled by adaptive pooling in model

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
- EXCLUDE = Remove from dataset (revision cases, poor quality, etc.)

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

Given **high-end GPU availability**, optimize for **maximum accuracy**.

#### **Architecture: Multi-Stream 3D CNN with Attention**

**Model Choice: 3D ResNet-50 or 3D EfficientNet-B4 Backbone**

With high-end GPU, we can use true 3D CNNs (superior to 2.5D for volumetric data).

```python
TemporalBoneClassifier_v2:
  
  Dual-Stream Architecture:
  
  Stream 1: Axial Volume Processing
    Input: (variable 30-40, 224, 224, 1) - variable-length axial slices
    ├─ 3D ResNet-50 backbone (pretrained on Kinetics-700 or MedicalNet)
    ├─ 3D Adaptive Average Pooling to handle variable slice counts
    │   └─ AdaptiveAvgPool3d(output_size=(8, 7, 7))
    │   └─ Converts (30-40, 28, 28) → (8, 7, 7) fixed size
    ├─ 3D spatial attention modules (capture relevant regions)
    └─ Feature vector: 2048 dimensions
  
  Stream 2: Coronal Volume Processing
    Input: (variable 20-30, 224, 224, 1) - variable-length coronal slices
    ├─ 3D ResNet-50 backbone (shared weights with Stream 1)
    ├─ Specialized for facial nerve canal visualization
    ├─ 3D Adaptive Average Pooling (same as Stream 1)
    ├─ 3D spatial attention modules
    └─ Feature vector: 2048 dimensions
  
  Feature Fusion:
    ├─ Concatenate axial and coronal features: 4096-dim
    ├─ Multi-head self-attention layer (learn inter-stream relationships)
    ├─ Fully connected fusion: FC(4096 → 1024) → ReLU → Dropout(0.4)
    └─ Shared representation: 1024 dimensions
  
  Classification Heads (Separate for Each Pathology):
    
    Cholesteatoma Head:
      ├─ FC(1024 → 512) → ReLU → Dropout(0.3)
      ├─ FC(512 → 256) → ReLU → Dropout(0.3)
      ├─ FC(256 → 1) → Sigmoid
      └─ Output: P(cholesteatoma) ∈ [0,1]
    
    Ossicular Head:
      ├─ FC(1024 → 512) → ReLU → Dropout(0.3)
      ├─ FC(512 → 256) → ReLU → Dropout(0.3)
      ├─ FC(256 → 1) → Sigmoid
      └─ Output: P(ossicular_discontinuity) ∈ [0,1]
    
    Facial Nerve Head:
      ├─ Enhanced coronal stream weighting (facial nerve better on coronal)
      ├─ FC(1024 → 512) → ReLU → Dropout(0.3)
      ├─ FC(512 → 256) → ReLU → Dropout(0.3)
      ├─ FC(256 → 1) → Sigmoid
      └─ Output: P(facial_dehiscence) ∈ [0,1]
```

**Key Architecture Features:**

**3D CNN Advantages:**
- Native volumetric processing (better than 2.5D stacking)
- Captures true 3D spatial relationships
- Better for subtle bone defects (facial nerve dehiscence)
- State-of-art for medical imaging

**Dual-Stream Design:**
- Axial stream: Primary view for cholesteatoma and ossicular chain
- Coronal stream: Superior for facial nerve canal and scutum
- Combined features leverage both perspectives

**Attention Mechanisms:**
- 3D spatial attention highlights relevant anatomical regions
- Reduces impact of irrelevant background structures
- Improves interpretability

**Separate Classification Heads:**
- Independent probability calibration per pathology
- Pathology-specific feature learning
- Better handles varying documentation rates (facial nerve)

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

### **Phase 6: Advanced Techniques (If Time/Resources Permit)**

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