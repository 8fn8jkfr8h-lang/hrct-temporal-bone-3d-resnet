"""
Phase 3: Dataset Stratification (Validation Script)
Temporal Bone HRCT Project

Simplified version for pipeline validation with limited dataset (24 patients):
- NO fixed test set (use all data for 5-fold CV)
- 2-label stratification (cholesteatoma + ossicular only)
- Patient-level grouping to prevent data leakage from bilateral ears

Usage:
    python -m pipeline.phase3_dataset_stratification_validation \
        --roi_dir roi_extracted \
        --labels_csv labels.csv \
        --output_dir dataset_splits_validation \
        --cv_folds 5 \
        --random_seed 69
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_and_validate_labels(labels_path: Path) -> pd.DataFrame:
    """
    Load labels CSV and validate required columns.
    
    Args:
        labels_path: Path to labels.csv file
        
    Returns:
        Validated DataFrame with included samples only
    """
    logger.info(f"Loading labels from {labels_path}")
    
    # Required columns
    required_cols = ['patient_id', 'ear', 'cholesteatoma', 'ossicular_discontinuity', 
                     'exclusion_status']
    
    df = pd.read_csv(labels_path)
    
    # Validate schema
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check for missing patient_id or ear
    if df['patient_id'].isna().any():
        raise ValueError("Found rows with missing patient_id")
    if df['ear'].isna().any():
        raise ValueError("Found rows with missing ear")
    
    # Apply exclusion filter
    included_df = df[df['exclusion_status'] == 'include'].copy()
    excluded_count = len(df) - len(included_df)
    
    logger.info(f"Total samples: {len(df)}, Included: {len(included_df)}, Excluded: {excluded_count}")
    
    # Normalize ear column (L/R to left/right for consistency)
    included_df['ear_normalized'] = included_df['ear'].map({'L': 'left', 'R': 'right'})
    if included_df['ear_normalized'].isna().any():
        # Handle if already 'left'/'right' format
        mask = included_df['ear_normalized'].isna()
        included_df.loc[mask, 'ear_normalized'] = included_df.loc[mask, 'ear'].str.lower()
    
    # Create ear_id
    included_df['ear_id'] = included_df['patient_id'] + '_' + included_df['ear']
    
    return included_df


def validate_roi_availability(labels_df: pd.DataFrame, roi_dir: Path) -> pd.DataFrame:
    """
    Check which samples have corresponding ROI files.
    
    Args:
        labels_df: DataFrame with label information
        roi_dir: Path to ROI extracted directory
        
    Returns:
        DataFrame with added 'roi_exists' column, filtered to only samples with ROIs
    """
    logger.info(f"Validating ROI availability in {roi_dir}")
    
    roi_exists = []
    for _, row in labels_df.iterrows():
        roi_path = roi_dir / row['patient_id'] / row['ear_normalized'] / 'axial_roi.npy'
        roi_exists.append(roi_path.exists())
    
    labels_df = labels_df.copy()
    labels_df['roi_exists'] = roi_exists
    
    missing_rois = labels_df[~labels_df['roi_exists']]
    if len(missing_rois) > 0:
        logger.warning(f"Missing ROIs for {len(missing_rois)} samples:")
        for _, row in missing_rois.iterrows():
            logger.warning(f"  - {row['ear_id']}")
    
    # Filter to only samples with ROIs
    valid_df = labels_df[labels_df['roi_exists']].copy()
    logger.info(f"Samples with valid ROIs: {len(valid_df)}")
    
    return valid_df


def create_patient_level_labels(labels_df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Create patient-level label matrix for stratification.
    Uses OR logic for bilateral cases (if either ear is positive, patient is positive).
    
    Args:
        labels_df: DataFrame with ear-level labels
        
    Returns:
        Tuple of:
        - Patient-level DataFrame with aggregated labels
        - Label matrix (n_patients, 2) for stratification
    """
    logger.info("Creating patient-level labels with OR logic for bilateral cases")
    
    # Group by patient and aggregate labels
    patient_labels = labels_df.groupby('patient_id').agg({
        'cholesteatoma': 'max',  # OR logic: max of 0/1 = 1 if any positive
        'ossicular_discontinuity': 'max',
        'ear_id': list,  # Keep track of which ears belong to patient
        'ear_normalized': list
    }).reset_index()
    
    patient_labels.rename(columns={'ear_id': 'ear_ids', 'ear_normalized': 'ears'}, inplace=True)
    
    # Fill NaN with 0 for stratification (treat unknown as negative)
    patient_labels['cholesteatoma'] = patient_labels['cholesteatoma'].fillna(0).astype(int)
    patient_labels['ossicular_discontinuity'] = patient_labels['ossicular_discontinuity'].fillna(0).astype(int)
    
    # Create label matrix for stratification [cholesteatoma, ossicular]
    label_matrix = patient_labels[['cholesteatoma', 'ossicular_discontinuity']].values
    
    logger.info(f"Patient-level statistics:")
    logger.info(f"  Total patients: {len(patient_labels)}")
    logger.info(f"  Cholesteatoma positive: {label_matrix[:, 0].sum()}")
    logger.info(f"  Ossicular discontinuity positive: {label_matrix[:, 1].sum()}")
    
    return patient_labels, label_matrix


def create_cv_splits(
    patient_labels: pd.DataFrame,
    label_matrix: np.ndarray,
    n_folds: int = 5,
    random_seed: int = 69
) -> List[Dict[str, Any]]:
    """
    Create stratified k-fold cross-validation splits at patient level.
    
    Args:
        patient_labels: Patient-level DataFrame
        label_matrix: Label matrix for stratification
        n_folds: Number of CV folds
        random_seed: Random seed for reproducibility
        
    Returns:
        List of fold dictionaries with train/val splits
    """
    logger.info(f"Creating {n_folds}-fold CV splits with seed {random_seed}")
    
    mskf = MultilabelStratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    
    patient_ids = patient_labels['patient_id'].values
    folds = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(mskf.split(patient_ids, label_matrix)):
        train_patient_ids = patient_ids[train_idx].tolist()
        val_patient_ids = patient_ids[val_idx].tolist()
        
        # Get ear IDs for train and val
        train_ear_ids = []
        val_ear_ids = []
        
        for pid in train_patient_ids:
            ear_ids = patient_labels[patient_labels['patient_id'] == pid]['ear_ids'].values[0]
            train_ear_ids.extend(ear_ids)
        
        for pid in val_patient_ids:
            ear_ids = patient_labels[patient_labels['patient_id'] == pid]['ear_ids'].values[0]
            val_ear_ids.extend(ear_ids)
        
        # Calculate class distribution
        train_labels = label_matrix[train_idx]
        val_labels = label_matrix[val_idx]
        
        fold_data = {
            'fold': fold_idx,
            'train_patient_ids': train_patient_ids,
            'val_patient_ids': val_patient_ids,
            'train_ear_ids': train_ear_ids,
            'val_ear_ids': val_ear_ids,
            'n_train_patients': len(train_patient_ids),
            'n_val_patients': len(val_patient_ids),
            'n_train_ears': len(train_ear_ids),
            'n_val_ears': len(val_ear_ids),
            'train_distribution': {
                'cholesteatoma': {'positive': int(train_labels[:, 0].sum()), 
                                  'negative': int((1 - train_labels[:, 0]).sum())},
                'ossicular_discontinuity': {'positive': int(train_labels[:, 1].sum()),
                                            'negative': int((1 - train_labels[:, 1]).sum())}
            },
            'val_distribution': {
                'cholesteatoma': {'positive': int(val_labels[:, 0].sum()),
                                  'negative': int((1 - val_labels[:, 0]).sum())},
                'ossicular_discontinuity': {'positive': int(val_labels[:, 1].sum()),
                                            'negative': int((1 - val_labels[:, 1]).sum())}
            }
        }
        
        folds.append(fold_data)
        
        logger.info(f"Fold {fold_idx}: Train {len(train_patient_ids)} patients ({len(train_ear_ids)} ears), "
                   f"Val {len(val_patient_ids)} patients ({len(val_ear_ids)} ears)")
    
    return folds


def validate_splits(folds: List[Dict], patient_labels: pd.DataFrame) -> bool:
    """
    Validate that splits are correct (no data leakage, all patients covered).
    
    Args:
        folds: List of fold dictionaries
        patient_labels: Patient-level DataFrame
        
    Returns:
        True if validation passes
    """
    logger.info("Validating split integrity...")
    
    all_valid = True
    all_patients = set(patient_labels['patient_id'])
    
    for fold in folds:
        fold_idx = fold['fold']
        train_set = set(fold['train_patient_ids'])
        val_set = set(fold['val_patient_ids'])
        
        # Check no overlap
        overlap = train_set & val_set
        if overlap:
            logger.error(f"Fold {fold_idx}: Overlap between train and val: {overlap}")
            all_valid = False
        
        # Check all patients covered
        covered = train_set | val_set
        missing = all_patients - covered
        if missing:
            logger.error(f"Fold {fold_idx}: Missing patients: {missing}")
            all_valid = False
        
        # Check no extra patients
        extra = covered - all_patients
        if extra:
            logger.error(f"Fold {fold_idx}: Extra patients not in dataset: {extra}")
            all_valid = False
    
    if all_valid:
        logger.info("All validation checks passed!")
    
    return all_valid


def print_summary(folds: List[Dict], patient_labels: pd.DataFrame, label_matrix: np.ndarray):
    """Print summary statistics table."""
    print("\n" + "=" * 80)
    print("DATASET STRATIFICATION SUMMARY (VALIDATION)")
    print("=" * 80)
    
    print(f"\nTotal Patients: {len(patient_labels)}")
    total_ears = sum(len(ears) for ears in patient_labels['ear_ids'])
    print(f"Total Ears: {total_ears}")
    
    print(f"\nOverall Class Distribution (Patient-Level):")
    print(f"  Cholesteatoma:           {label_matrix[:, 0].sum():3d} positive / {len(label_matrix):3d} total ({100*label_matrix[:, 0].mean():.1f}%)")
    print(f"  Ossicular Discontinuity: {label_matrix[:, 1].sum():3d} positive / {len(label_matrix):3d} total ({100*label_matrix[:, 1].mean():.1f}%)")
    
    print(f"\nFold Summary:")
    print("-" * 80)
    print(f"{'Fold':<6} {'Train Pts':<12} {'Val Pts':<10} {'Train Chole+':<14} {'Val Chole+':<12} {'Train Ossic+':<14} {'Val Ossic+':<12}")
    print("-" * 80)
    
    for fold in folds:
        train_chole = fold['train_distribution']['cholesteatoma']['positive']
        val_chole = fold['val_distribution']['cholesteatoma']['positive']
        train_ossic = fold['train_distribution']['ossicular_discontinuity']['positive']
        val_ossic = fold['val_distribution']['ossicular_discontinuity']['positive']
        
        print(f"{fold['fold']:<6} {fold['n_train_patients']:<12} {fold['n_val_patients']:<10} "
              f"{train_chole:<14} {val_chole:<12} {train_ossic:<14} {val_ossic:<12}")
    
    print("-" * 80)
    print()


def save_outputs(
    folds: List[Dict],
    patient_labels: pd.DataFrame,
    label_matrix: np.ndarray,
    output_dir: Path,
    config: Dict[str, Any]
):
    """Save fold JSONs and metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save each fold
    for fold in folds:
        fold_path = output_dir / f"fold_{fold['fold']}.json"
        with open(fold_path, 'w') as f:
            json.dump(fold, f, indent=2)
        logger.info(f"Saved {fold_path}")
    
    # Save metadata
    total_ears = sum(len(ears) for ears in patient_labels['ear_ids'])
    
    metadata = {
        'random_seed': config['random_seed'],
        'n_total_patients': len(patient_labels),
        'n_total_ears': total_ears,
        'cv_folds': config['cv_folds'],
        'stratification_labels': ['cholesteatoma', 'ossicular_discontinuity'],
        'script_version': 'validation',
        'overall_distribution': {
            'cholesteatoma': {
                'positive': int(label_matrix[:, 0].sum()),
                'negative': int((1 - label_matrix[:, 0]).sum())
            },
            'ossicular_discontinuity': {
                'positive': int(label_matrix[:, 1].sum()),
                'negative': int((1 - label_matrix[:, 1]).sum())
            }
        },
        'timestamp': datetime.now().isoformat()
    }
    
    metadata_path = output_dir / 'split_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved {metadata_path}")
    
    # Save summary text
    summary_path = output_dir / 'split_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("DATASET STRATIFICATION SUMMARY (VALIDATION)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total Patients: {len(patient_labels)}\n")
        f.write(f"Total Ears: {total_ears}\n")
        f.write(f"CV Folds: {config['cv_folds']}\n")
        f.write(f"Random Seed: {config['random_seed']}\n")
        f.write(f"Generated: {metadata['timestamp']}\n")
    logger.info(f"Saved {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Phase 3: Dataset Stratification (Validation Script)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--roi_dir', type=str, default='roi_extracted',
                        help='Directory containing extracted ROIs')
    parser.add_argument('--labels_csv', type=str, default='labels.csv',
                        help='Path to labels CSV file')
    parser.add_argument('--output_dir', type=str, default='dataset_splits_validation',
                        help='Output directory for split files')
    parser.add_argument('--cv_folds', type=int, default=5,
                        help='Number of cross-validation folds')
    parser.add_argument('--random_seed', type=int, default=69,
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Convert to Path objects
    roi_dir = Path(args.roi_dir)
    labels_path = Path(args.labels_csv)
    output_dir = Path(args.output_dir)
    
    config = {
        'roi_dir': str(roi_dir),
        'labels_csv': str(labels_path),
        'output_dir': str(output_dir),
        'cv_folds': args.cv_folds,
        'random_seed': args.random_seed
    }
    
    logger.info("=" * 60)
    logger.info("Phase 3: Dataset Stratification (Validation)")
    logger.info("=" * 60)
    
    # Step 1: Load and validate labels
    labels_df = load_and_validate_labels(labels_path)
    
    # Step 2: Validate ROI availability
    labels_df = validate_roi_availability(labels_df, roi_dir)
    
    if len(labels_df) == 0:
        logger.error("No valid samples found with ROIs. Exiting.")
        return
    
    # Step 3: Create patient-level labels
    patient_labels, label_matrix = create_patient_level_labels(labels_df)
    
    # Step 4: Create CV splits
    folds = create_cv_splits(patient_labels, label_matrix, args.cv_folds, args.random_seed)
    
    # Step 5: Validate splits
    if not validate_splits(folds, patient_labels):
        logger.error("Split validation failed!")
        return
    
    # Step 6: Print summary
    print_summary(folds, patient_labels, label_matrix)
    
    # Step 7: Save outputs
    save_outputs(folds, patient_labels, label_matrix, output_dir, config)
    
    logger.info("Phase 3 (Validation) completed successfully!")


if __name__ == '__main__':
    main()
