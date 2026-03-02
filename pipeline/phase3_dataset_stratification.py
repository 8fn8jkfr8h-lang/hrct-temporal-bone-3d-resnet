"""
Phase 3: Dataset Stratification (Production Script)
Temporal Bone HRCT Project

Full-featured version for production dataset (100-120 patients):
- Fixed test set allocation (15-20%)
- 4-label stratification (cholesteatoma + ossicular + facial_presence + lscc_presence)
- Patient-level grouping to prevent data leakage from bilateral ears
- 5-fold cross-validation on remaining data

Usage:
    python -m pipeline.phase3_dataset_stratification \
        --roi_dir roi_extracted \
        --labels_csv labels.csv \
        --output_dir dataset_splits \
        --test_ratio 0.18 \
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
from scipy.stats import chi2_contingency
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit

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
    
    # Required columns for production (includes facial and LSCC dehiscence)
    required_cols = ['patient_id', 'ear', 'cholesteatoma', 'ossicular_discontinuity', 
                     'facial_dehiscence', 'lscc_dehiscence', 'exclusion_status']
    
    df = pd.read_csv(labels_path)
    df = df.dropna(how='all').copy()
    
    # Validate schema
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Drop malformed rows (common when CSV has trailing blank lines/columns)
    missing_id_or_ear = df['patient_id'].isna() | df['ear'].isna()
    if missing_id_or_ear.any():
        n_dropped = int(missing_id_or_ear.sum())
        logger.warning(f"Dropping {n_dropped} rows with missing patient_id/ear")
        df = df.loc[~missing_id_or_ear].copy()
    
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
    
    # Create facial_nerve_presence indicator (1 if value exists, 0 if NULL)
    included_df['facial_nerve_presence'] = (~included_df['facial_dehiscence'].isna()).astype(int)
    # Create LSCC presence indicator (1 if value exists, 0 if NULL)
    included_df['lscc_presence'] = (~included_df['lscc_dehiscence'].isna()).astype(int)
    
    return included_df


def validate_roi_availability(labels_df: pd.DataFrame, roi_dir: Path) -> pd.DataFrame:
    """
    Check which samples have corresponding ROI files.
    
    Args:
        labels_df: DataFrame with label information
        roi_dir: Path to ROI extracted directory
        
    Returns:
        DataFrame filtered to only samples with ROIs
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
    Uses OR logic for bilateral cases.
    
    Args:
        labels_df: DataFrame with ear-level labels
        
    Returns:
        Tuple of:
        - Patient-level DataFrame with aggregated labels
        - Label matrix (n_patients, 4) for stratification [chole, ossic, facial_presence, lscc_presence]
    """
    logger.info("Creating patient-level labels with OR logic for bilateral cases")
    
    # Group by patient and aggregate labels
    patient_labels = labels_df.groupby('patient_id').agg({
        'cholesteatoma': 'max',
        'ossicular_discontinuity': 'max',
        'facial_dehiscence': 'max',
        'facial_nerve_presence': 'max',
        'lscc_dehiscence': 'max',
        'lscc_presence': 'max',
        'ear_id': list,
        'ear_normalized': list
    }).reset_index()
    
    patient_labels.rename(columns={'ear_id': 'ear_ids', 'ear_normalized': 'ears'}, inplace=True)
    
    # Fill NaN with 0 for stratification
    patient_labels['cholesteatoma'] = patient_labels['cholesteatoma'].fillna(0).astype(int)
    patient_labels['ossicular_discontinuity'] = patient_labels['ossicular_discontinuity'].fillna(0).astype(int)
    patient_labels['facial_nerve_presence'] = patient_labels['facial_nerve_presence'].fillna(0).astype(int)
    patient_labels['lscc_presence'] = patient_labels['lscc_presence'].fillna(0).astype(int)

    # Create label matrix for stratification [cholesteatoma, ossicular, facial_presence, lscc_presence]
    label_matrix = patient_labels[
        ['cholesteatoma', 'ossicular_discontinuity', 'facial_nerve_presence', 'lscc_presence']
    ].values
    
    logger.info(f"Patient-level statistics:")
    logger.info(f"  Total patients: {len(patient_labels)}")
    logger.info(f"  Cholesteatoma positive: {label_matrix[:, 0].sum()}")
    logger.info(f"  Ossicular discontinuity positive: {label_matrix[:, 1].sum()}")
    logger.info(f"  Facial nerve data present: {label_matrix[:, 2].sum()}")
    logger.info(f"  LSCC data present: {label_matrix[:, 3].sum()}")
    
    return patient_labels, label_matrix


def allocate_test_set(
    patient_labels: pd.DataFrame,
    label_matrix: np.ndarray,
    test_ratio: float = 0.18,
    random_seed: int = 69
) -> Tuple[List[str], List[str], np.ndarray, np.ndarray]:
    """
    Allocate fixed test set using stratified sampling.
    
    Args:
        patient_labels: Patient-level DataFrame
        label_matrix: Label matrix for stratification
        test_ratio: Fraction of patients for test set
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (trainval_patient_ids, test_patient_ids, trainval_labels, test_labels)
    """
    logger.info(f"Allocating test set ({test_ratio*100:.0f}% of patients)")
    
    patient_ids = patient_labels['patient_id'].values
    
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=random_seed)
    
    for trainval_idx, test_idx in msss.split(patient_ids, label_matrix):
        trainval_patient_ids = patient_ids[trainval_idx].tolist()
        test_patient_ids = patient_ids[test_idx].tolist()
        trainval_labels = label_matrix[trainval_idx]
        test_labels = label_matrix[test_idx]
    
    logger.info(f"  Train/Val patients: {len(trainval_patient_ids)}")
    logger.info(f"  Test patients: {len(test_patient_ids)}")
    
    return trainval_patient_ids, test_patient_ids, trainval_labels, test_labels


def create_test_set_json(
    test_patient_ids: List[str],
    patient_labels: pd.DataFrame,
    labels_df: pd.DataFrame
) -> Dict[str, Any]:
    """Create test set JSON structure."""
    
    # Get ear IDs for test patients
    test_ear_ids = []
    for pid in test_patient_ids:
        ear_ids = patient_labels[patient_labels['patient_id'] == pid]['ear_ids'].values[0]
        test_ear_ids.extend(ear_ids)
    
    # Calculate class distribution
    test_ears_df = labels_df[labels_df['patient_id'].isin(test_patient_ids)]
    
    def count_distribution(series):
        positive = int((series == 1).sum())
        negative = int((series == 0).sum())
        null = int(series.isna().sum())
        return {'positive': positive, 'negative': negative, 'null': null} if null > 0 else {'positive': positive, 'negative': negative}
    
    return {
        'patient_ids': test_patient_ids,
        'ear_ids': test_ear_ids,
        'n_patients': len(test_patient_ids),
        'n_ears': len(test_ear_ids),
        'class_distribution': {
            'cholesteatoma': count_distribution(test_ears_df['cholesteatoma']),
            'ossicular_discontinuity': count_distribution(test_ears_df['ossicular_discontinuity']),
            'facial_nerve_dehiscence': count_distribution(test_ears_df['facial_dehiscence']),
            'lscc_dehiscence': count_distribution(test_ears_df['lscc_dehiscence'])
        }
    }


def create_cv_splits(
    trainval_patient_ids: List[str],
    trainval_labels: np.ndarray,
    patient_labels: pd.DataFrame,
    n_folds: int = 5,
    random_seed: int = 69
) -> List[Dict[str, Any]]:
    """
    Create stratified k-fold cross-validation splits.
    
    Args:
        trainval_patient_ids: Patient IDs for train/val (excluding test set)
        trainval_labels: Label matrix for stratification
        patient_labels: Full patient-level DataFrame
        n_folds: Number of CV folds
        random_seed: Random seed for reproducibility
        
    Returns:
        List of fold dictionaries
    """
    logger.info(f"Creating {n_folds}-fold CV splits on train/val data")
    
    mskf = MultilabelStratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
    
    trainval_patient_ids = np.array(trainval_patient_ids)
    folds = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(mskf.split(trainval_patient_ids, trainval_labels)):
        train_patient_ids = trainval_patient_ids[train_idx].tolist()
        val_patient_ids = trainval_patient_ids[val_idx].tolist()
        
        # Get ear IDs
        train_ear_ids = []
        val_ear_ids = []
        
        for pid in train_patient_ids:
            ear_ids = patient_labels[patient_labels['patient_id'] == pid]['ear_ids'].values[0]
            train_ear_ids.extend(ear_ids)
        
        for pid in val_patient_ids:
            ear_ids = patient_labels[patient_labels['patient_id'] == pid]['ear_ids'].values[0]
            val_ear_ids.extend(ear_ids)
        
        # Calculate class distribution
        train_labels = trainval_labels[train_idx]
        val_labels = trainval_labels[val_idx]
        
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
                                            'negative': int((1 - train_labels[:, 1]).sum())},
                'facial_nerve_presence': {'positive': int(train_labels[:, 2].sum()),
                                          'negative': int((1 - train_labels[:, 2]).sum())},
                'lscc_presence': {'positive': int(train_labels[:, 3].sum()),
                                  'negative': int((1 - train_labels[:, 3]).sum())}
            },
            'val_distribution': {
                'cholesteatoma': {'positive': int(val_labels[:, 0].sum()),
                                  'negative': int((1 - val_labels[:, 0]).sum())},
                'ossicular_discontinuity': {'positive': int(val_labels[:, 1].sum()),
                                            'negative': int((1 - val_labels[:, 1]).sum())},
                'facial_nerve_presence': {'positive': int(val_labels[:, 2].sum()),
                                          'negative': int((1 - val_labels[:, 2]).sum())},
                'lscc_presence': {'positive': int(val_labels[:, 3].sum()),
                                  'negative': int((1 - val_labels[:, 3]).sum())}
            }
        }
        
        folds.append(fold_data)
        
        logger.info(f"Fold {fold_idx}: Train {len(train_patient_ids)} patients ({len(train_ear_ids)} ears), "
                   f"Val {len(val_patient_ids)} patients ({len(val_ear_ids)} ears)")
    
    return folds


def check_class_balance(folds: List[Dict], label_name: str = 'cholesteatoma') -> Tuple[float, float]:
    """
    Check class distribution balance across folds using chi-square test.
    
    Returns:
        Tuple of (chi2_statistic, p_value)
    """
    # Build contingency table: rows = folds, cols = [positive, negative]
    observed = []
    for fold in folds:
        dist = fold['val_distribution'][label_name]
        observed.append([dist['positive'], dist['negative']])
    
    observed = np.array(observed)
    
    # Only run test if we have variation
    if observed.sum(axis=0).min() == 0:
        return 0.0, 1.0
    
    chi2, p_value, dof, expected = chi2_contingency(observed)
    return chi2, p_value


def validate_splits(
    folds: List[Dict],
    test_set: Dict,
    patient_labels: pd.DataFrame
) -> bool:
    """Validate split integrity."""
    logger.info("Validating split integrity...")
    
    all_valid = True
    all_patients = set(patient_labels['patient_id'])
    test_patients = set(test_set['patient_ids'])
    
    # Check test set doesn't overlap with any train/val
    for fold in folds:
        train_set = set(fold['train_patient_ids'])
        val_set = set(fold['val_patient_ids'])
        
        # No overlap with test
        if train_set & test_patients:
            logger.error(f"Fold {fold['fold']}: Train overlaps with test set!")
            all_valid = False
        if val_set & test_patients:
            logger.error(f"Fold {fold['fold']}: Val overlaps with test set!")
            all_valid = False
        
        # No overlap between train and val
        if train_set & val_set:
            logger.error(f"Fold {fold['fold']}: Train/val overlap!")
            all_valid = False
        
        # All trainval patients covered
        fold_patients = train_set | val_set
        trainval_patients = all_patients - test_patients
        if fold_patients != trainval_patients:
            logger.error(f"Fold {fold['fold']}: Missing or extra patients in train/val!")
            all_valid = False
    
    # Chi-square test for balance
    chi2, p_value = check_class_balance(folds, 'cholesteatoma')
    logger.info(f"Chi-square test for cholesteatoma balance: chi2={chi2:.2f}, p={p_value:.4f}")
    if p_value < 0.05:
        logger.warning("Significant class imbalance across folds detected!")
    
    if all_valid:
        logger.info("All validation checks passed!")
    
    return all_valid


def print_summary(
    folds: List[Dict],
    test_set: Dict,
    patient_labels: pd.DataFrame,
    label_matrix: np.ndarray
):
    """Print summary statistics table."""
    print("\n" + "=" * 90)
    print("DATASET STRATIFICATION SUMMARY (PRODUCTION)")
    print("=" * 90)
    
    print(f"\nTotal Patients: {len(patient_labels)}")
    total_ears = sum(len(ears) for ears in patient_labels['ear_ids'])
    print(f"Total Ears: {total_ears}")
    
    print(f"\nTest Set: {test_set['n_patients']} patients ({test_set['n_ears']} ears)")
    print(f"Train/Val: {len(patient_labels) - test_set['n_patients']} patients")
    
    print(f"\nOverall Class Distribution (Patient-Level):")
    print(f"  Cholesteatoma:           {label_matrix[:, 0].sum():3d} positive / {len(label_matrix):3d} total ({100*label_matrix[:, 0].mean():.1f}%)")
    print(f"  Ossicular Discontinuity: {label_matrix[:, 1].sum():3d} positive / {len(label_matrix):3d} total ({100*label_matrix[:, 1].mean():.1f}%)")
    print(f"  Facial Nerve Data:       {label_matrix[:, 2].sum():3d} present / {len(label_matrix):3d} total ({100*label_matrix[:, 2].mean():.1f}%)")
    print(f"  LSCC Data:               {label_matrix[:, 3].sum():3d} present / {len(label_matrix):3d} total ({100*label_matrix[:, 3].mean():.1f}%)")
    
    print(f"\nTest Set Class Distribution:")
    for label, dist in test_set['class_distribution'].items():
        pos = dist['positive']
        neg = dist['negative']
        null = dist.get('null', 0)
        print(f"  {label}: +{pos} / -{neg} / null:{null}")
    
    print(f"\nFold Summary (Train/Val only, excludes test set):")
    print("-" * 90)
    print(f"{'Fold':<6} {'Train Pts':<12} {'Val Pts':<10} {'Train Chole+':<14} {'Val Chole+':<12}")
    print("-" * 90)
    
    for fold in folds:
        train_chole = fold['train_distribution']['cholesteatoma']['positive']
        val_chole = fold['val_distribution']['cholesteatoma']['positive']
        
        print(f"{fold['fold']:<6} {fold['n_train_patients']:<12} {fold['n_val_patients']:<10} "
              f"{train_chole:<14} {val_chole:<12}")
    
    print("-" * 90)
    print()


def save_outputs(
    folds: List[Dict],
    test_set: Dict,
    patient_labels: pd.DataFrame,
    label_matrix: np.ndarray,
    output_dir: Path,
    config: Dict[str, Any]
):
    """Save all output files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save test set
    test_path = output_dir / 'test_set.json'
    with open(test_path, 'w') as f:
        json.dump(test_set, f, indent=2)
    logger.info(f"Saved {test_path}")
    
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
        'test_percentage': round(100 * test_set['n_patients'] / len(patient_labels), 1),
        'cv_folds': config['cv_folds'],
        'stratification_labels': ['cholesteatoma', 'ossicular_discontinuity', 'facial_nerve_presence', 'lscc_presence'],
        'script_version': 'production',
        'exclusion_criteria': {
            'excluded_patients': config.get('excluded_count', 0)
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
        f.write("DATASET STRATIFICATION SUMMARY (PRODUCTION)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total Patients: {len(patient_labels)}\n")
        f.write(f"Total Ears: {total_ears}\n")
        f.write(f"Test Set: {test_set['n_patients']} patients ({test_set['n_ears']} ears)\n")
        f.write(f"CV Folds: {config['cv_folds']}\n")
        f.write(f"Random Seed: {config['random_seed']}\n")
        f.write(f"Generated: {metadata['timestamp']}\n")
    logger.info(f"Saved {summary_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Phase 3: Dataset Stratification (Production Script)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--roi_dir', type=str, default='roi_extracted',
                        help='Directory containing extracted ROIs')
    parser.add_argument('--labels_csv', type=str, default='labels.csv',
                        help='Path to labels CSV file')
    parser.add_argument('--output_dir', type=str, default='dataset_splits',
                        help='Output directory for split files')
    parser.add_argument('--test_ratio', type=float, default=0.18,
                        help='Fraction of patients for fixed test set')
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
        'test_ratio': args.test_ratio,
        'cv_folds': args.cv_folds,
        'random_seed': args.random_seed
    }
    
    logger.info("=" * 60)
    logger.info("Phase 3: Dataset Stratification (Production)")
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
    
    # Step 4: Allocate test set
    trainval_ids, test_ids, trainval_labels, test_labels = allocate_test_set(
        patient_labels, label_matrix, args.test_ratio, args.random_seed
    )
    
    # Step 5: Create test set JSON
    test_set = create_test_set_json(test_ids, patient_labels, labels_df)
    
    # Step 6: Create CV splits on remaining data
    folds = create_cv_splits(trainval_ids, trainval_labels, patient_labels, 
                            args.cv_folds, args.random_seed)
    
    # Step 7: Validate splits
    if not validate_splits(folds, test_set, patient_labels):
        logger.error("Split validation failed!")
        return
    
    # Step 8: Print summary
    print_summary(folds, test_set, patient_labels, label_matrix)
    
    # Step 9: Save outputs
    save_outputs(folds, test_set, patient_labels, label_matrix, output_dir, config)
    
    logger.info("Phase 3 (Production) completed successfully!")


if __name__ == '__main__':
    main()
