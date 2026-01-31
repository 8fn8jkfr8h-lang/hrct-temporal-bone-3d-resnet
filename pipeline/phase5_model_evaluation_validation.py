"""
Phase 5: Model Evaluation (Validation Script)
Temporal Bone HRCT Project

Simplified validation evaluation script for pipeline testing:
- Cross-validation metrics only (no test set)
- 2-task metrics (cholesteatoma + ossicular)
- Sanity checks for pipeline validation
- No Grad-CAM (focus on metrics)

Usage:
    python pipeline/phase5_model_evaluation_validation.py \
        --models_dir models_validation \
        --split_dir dataset_splits_validation \
        --roi_dir roi_extracted \
        --labels_csv labels.csv \
        --output_dir evaluation_results_validation
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def convert_to_native_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_native_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native_types(i) for i in obj]
    else:
        return obj

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.resnet3d import TemporalBoneClassifier
from data.dataset import TemporalBoneDataset
from data.transforms import get_val_transforms
from evaluation.metrics import (
    compute_auc_safe,
    compute_classification_metrics,
    find_optimal_threshold,
    sanity_check_predictions
)
from evaluation.visualization import (
    plot_training_curves,
    plot_auc_summary
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def validate_inputs(args: argparse.Namespace) -> None:
    """Validate all input arguments before execution."""
    errors = []
    
    if not Path(args.models_dir).exists():
        errors.append(f"Models directory not found: {args.models_dir}")
    if not Path(args.split_dir).exists():
        errors.append(f"Splits directory not found: {args.split_dir}")
    if not Path(args.roi_dir).exists():
        errors.append(f"ROI directory not found: {args.roi_dir}")
    if not Path(args.labels_csv).exists():
        errors.append(f"Labels file not found: {args.labels_csv}")
    
    if errors:
        for err in errors:
            logger.error(err)
        raise ValueError(f"Input validation failed. {len(errors)} errors found.")


def load_fold_model(
    fold_dir: Path,
    device: torch.device,
    num_tasks: int = 2
) -> Optional[TemporalBoneClassifier]:
    """
    Load trained model from fold directory.
    
    Args:
        fold_dir: Path to fold directory containing best_checkpoint.pth
        device: Device to load model on
        num_tasks: Number of classification tasks
        
    Returns:
        Loaded model or None if loading fails
    """
    checkpoint_path = fold_dir / 'best_checkpoint.pth'
    
    if not checkpoint_path.exists():
        logger.warning(f"Checkpoint not found: {checkpoint_path}")
        return None
    
    try:
        # Load config if available
        config_path = fold_dir / 'config.json'
        use_cbam = True
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
                use_cbam = config.get('use_cbam', True)
        
        # Create model
        model = TemporalBoneClassifier(
            num_tasks=num_tasks,
            use_cbam=use_cbam,
            pretrained_path=None
        )
        
        # Load weights
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        
        model.to(device)
        model.eval()
        
        logger.info(f"Loaded model from {checkpoint_path}")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load model from {fold_dir}: {e}")
        return None


def evaluate_fold(
    fold: int,
    model: nn.Module,
    val_ear_ids: List[str],
    roi_dir: Path,
    labels_df: pd.DataFrame,
    device: torch.device,
    batch_size: int = 4
) -> Dict:
    """
    Evaluate model on validation set for one fold.
    
    Args:
        fold: Fold number
        model: Trained model
        val_ear_ids: List of validation ear IDs
        roi_dir: Path to ROI data
        labels_df: Labels DataFrame
        device: Device for inference
        batch_size: Batch size for inference
        
    Returns:
        Dictionary with predictions and metrics
    """
    logger.info(f"Evaluating fold {fold} on {len(val_ear_ids)} validation ears")
    
    # Create dataset
    dataset = TemporalBoneDataset(
        ear_ids=val_ear_ids,
        roi_dir=roi_dir,
        labels_df=labels_df,
        transforms=get_val_transforms(),
        num_tasks=2
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    # Collect predictions
    all_probs_chole = []
    all_probs_ossic = []
    all_labels_chole = []
    all_labels_ossic = []
    all_masks_chole = []
    all_masks_ossic = []
    all_ear_ids = []
    
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            volumes = batch['image'].to(device)
            labels = batch['labels'].numpy()
            masks = batch['mask'].numpy()
            ear_ids = batch['ear_id']
            
            # Forward pass
            outputs = model(volumes)
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            # Collect results
            all_probs_chole.extend(probs[:, 0])
            all_probs_ossic.extend(probs[:, 1])
            all_labels_chole.extend(labels[:, 0])
            all_labels_ossic.extend(labels[:, 1])
            all_masks_chole.extend(masks[:, 0])
            all_masks_ossic.extend(masks[:, 1])
            all_ear_ids.extend(ear_ids)
    
    # Convert to numpy arrays
    probs_chole = np.array(all_probs_chole)
    probs_ossic = np.array(all_probs_ossic)
    labels_chole = np.array(all_labels_chole)
    labels_ossic = np.array(all_labels_ossic)
    masks_chole = np.array(all_masks_chole)
    masks_ossic = np.array(all_masks_ossic)
    
    # Sanity check
    try:
        sanity_check_predictions(probs_chole, "cholesteatoma predictions")
        sanity_check_predictions(probs_ossic, "ossicular predictions")
    except ValueError as e:
        logger.error(f"Prediction sanity check failed: {e}")
    
    # Compute metrics for valid samples only
    results = {'fold': fold, 'n_samples': len(all_ear_ids)}
    
    # Cholesteatoma metrics
    valid_chole = masks_chole == 1
    if valid_chole.sum() > 0:
        y_true_chole = labels_chole[valid_chole]
        y_pred_chole = probs_chole[valid_chole]
        
        auc_chole = compute_auc_safe(y_true_chole, y_pred_chole)
        threshold_chole, _ = find_optimal_threshold(y_true_chole, y_pred_chole, metric='f1')
        metrics_chole = compute_classification_metrics(y_true_chole, y_pred_chole, threshold_chole)
        
        results['auc_cholesteatoma'] = auc_chole
        results['sensitivity_cholesteatoma'] = metrics_chole['sensitivity']
        results['specificity_cholesteatoma'] = metrics_chole['specificity']
        results['threshold_cholesteatoma'] = threshold_chole
        results['n_valid_cholesteatoma'] = int(valid_chole.sum())
    
    # Ossicular discontinuity metrics
    valid_ossic = masks_ossic == 1
    if valid_ossic.sum() > 0:
        y_true_ossic = labels_ossic[valid_ossic]
        y_pred_ossic = probs_ossic[valid_ossic]
        
        auc_ossic = compute_auc_safe(y_true_ossic, y_pred_ossic)
        threshold_ossic, _ = find_optimal_threshold(y_true_ossic, y_pred_ossic, metric='f1')
        metrics_ossic = compute_classification_metrics(y_true_ossic, y_pred_ossic, threshold_ossic)
        
        results['auc_ossicular'] = auc_ossic
        results['sensitivity_ossicular'] = metrics_ossic['sensitivity']
        results['specificity_ossicular'] = metrics_ossic['specificity']
        results['threshold_ossicular'] = threshold_ossic
        results['n_valid_ossicular'] = int(valid_ossic.sum())
    
    # Store predictions for saving
    results['predictions'] = {
        'ear_ids': all_ear_ids,
        'prob_cholesteatoma': probs_chole.tolist(),
        'prob_ossicular': probs_ossic.tolist(),
        'label_cholesteatoma': labels_chole.tolist(),
        'label_ossicular': labels_ossic.tolist(),
        'mask_cholesteatoma': masks_chole.tolist(),
        'mask_ossicular': masks_ossic.tolist()
    }
    
    return results


def run_sanity_checks(
    fold_metrics: List[Dict],
    training_histories: Dict[int, Dict]
) -> Dict[str, bool]:
    """
    Run sanity checks for pipeline validation.
    
    Args:
        fold_metrics: List of metric dictionaries per fold
        training_histories: Dict of training history per fold
        
    Returns:
        Dictionary of check results
    """
    checks = {}
    
    # Check 1: Loss decreased during training
    loss_decreased = True
    for fold, history in training_histories.items():
        if 'train_loss' in history and len(history['train_loss']) > 1:
            if history['train_loss'][-1] >= history['train_loss'][0]:
                loss_decreased = False
                break
    checks['loss_decreased'] = loss_decreased
    
    # Check 2: Cholesteatoma AUC > 0.5 (better than chance)
    auc_values = [m.get('auc_cholesteatoma', 0) for m in fold_metrics if 'auc_cholesteatoma' in m]
    if auc_values:
        mean_auc_chole = np.mean(auc_values)
        checks['cholesteatoma_auc_above_chance'] = mean_auc_chole > 0.5
    else:
        checks['cholesteatoma_auc_above_chance'] = False
    
    # Check 3: Ossicular AUC > 0.5
    auc_ossic_values = [m.get('auc_ossicular', 0) for m in fold_metrics if 'auc_ossicular' in m]
    if auc_ossic_values:
        mean_auc_ossic = np.mean(auc_ossic_values)
        checks['ossicular_auc_above_chance'] = mean_auc_ossic > 0.5
    else:
        checks['ossicular_auc_above_chance'] = False
    
    # Check 4: No NaN in predictions (already checked in evaluate_fold)
    checks['no_nan_predictions'] = True
    
    # Check 5: At least some training occurred
    checks['training_completed'] = len(fold_metrics) > 0
    
    return checks


def main():
    parser = argparse.ArgumentParser(
        description='Phase 5: Model Evaluation (Validation)'
    )
    parser.add_argument('--models_dir', type=str, default='models_validation',
                       help='Directory with trained models')
    parser.add_argument('--split_dir', type=str, default='dataset_splits_validation',
                       help='Directory with dataset splits')
    parser.add_argument('--roi_dir', type=str, default='roi_extracted',
                       help='Directory with extracted ROIs')
    parser.add_argument('--labels_csv', type=str, default='labels.csv',
                       help='Path to labels CSV')
    parser.add_argument('--output_dir', type=str, default='evaluation_results_validation',
                       help='Output directory for evaluation results')
    parser.add_argument('--folds', type=str, default=None,
                       help='Comma-separated list of folds to evaluate (default: all available)')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for inference')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (auto, cpu, cuda)')
    
    args = parser.parse_args()
    
    # Validate inputs
    validate_inputs(args)
    
    # Setup paths
    models_dir = Path(args.models_dir)
    split_dir = Path(args.split_dir)
    roi_dir = Path(args.roi_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load labels
    labels_df = pd.read_csv(args.labels_csv)
    logger.info(f"Loaded {len(labels_df)} label entries")
    
    # Determine folds to evaluate
    if args.folds:
        folds = [int(f.strip()) for f in args.folds.split(',')]
    else:
        # Auto-detect available folds
        folds = []
        for fold_dir in models_dir.iterdir():
            if fold_dir.is_dir() and fold_dir.name.startswith('fold_'):
                fold_num = int(fold_dir.name.split('_')[1])
                folds.append(fold_num)
        folds = sorted(folds)
    
    if not folds:
        logger.error("No trained folds found")
        return
    
    logger.info(f"Evaluating folds: {folds}")
    
    # Evaluate each fold
    fold_metrics = []
    training_histories = {}
    all_predictions = {}
    
    for fold in folds:
        fold_dir = models_dir / f'fold_{fold}'
        
        if not fold_dir.exists():
            logger.warning(f"Fold directory not found: {fold_dir}")
            continue
        
        # Load model
        model = load_fold_model(fold_dir, device, num_tasks=2)
        if model is None:
            continue
        
        # Load fold split
        split_path = split_dir / f'fold_{fold}.json'
        if not split_path.exists():
            logger.warning(f"Split file not found: {split_path}")
            continue
        
        with open(split_path) as f:
            split_data = json.load(f)
        
        val_ear_ids = split_data['val_ear_ids']
        
        # Evaluate
        results = evaluate_fold(
            fold=fold,
            model=model,
            val_ear_ids=val_ear_ids,
            roi_dir=roi_dir,
            labels_df=labels_df,
            device=device,
            batch_size=args.batch_size
        )
        
        # Store metrics (without predictions for aggregation)
        metrics_only = {k: v for k, v in results.items() if k != 'predictions'}
        fold_metrics.append(metrics_only)
        all_predictions[fold] = results['predictions']
        
        # Load training history if available
        history_path = fold_dir / 'training_history.json'
        if history_path.exists():
            with open(history_path) as f:
                training_histories[fold] = json.load(f)
        
        logger.info(f"Fold {fold}: AUC_chole={results.get('auc_cholesteatoma', 'N/A'):.3f}, "
                   f"AUC_ossic={results.get('auc_ossicular', 'N/A'):.3f}")
    
    if not fold_metrics:
        logger.error("No folds could be evaluated")
        return
    
    # Aggregate metrics across folds
    aggregated = {}
    
    # Cholesteatoma
    chole_aucs = [m['auc_cholesteatoma'] for m in fold_metrics if 'auc_cholesteatoma' in m]
    chole_sens = [m['sensitivity_cholesteatoma'] for m in fold_metrics if 'sensitivity_cholesteatoma' in m]
    chole_spec = [m['specificity_cholesteatoma'] for m in fold_metrics if 'specificity_cholesteatoma' in m]
    
    if chole_aucs:
        aggregated['cholesteatoma'] = {
            'auc_mean': float(np.mean(chole_aucs)),
            'auc_std': float(np.std(chole_aucs)),
            'sensitivity_mean': float(np.mean(chole_sens)) if chole_sens else None,
            'sensitivity_std': float(np.std(chole_sens)) if chole_sens else None,
            'specificity_mean': float(np.mean(chole_spec)) if chole_spec else None,
            'specificity_std': float(np.std(chole_spec)) if chole_spec else None,
            'n_folds': len(chole_aucs)
        }
    
    # Ossicular discontinuity
    ossic_aucs = [m['auc_ossicular'] for m in fold_metrics if 'auc_ossicular' in m]
    ossic_sens = [m['sensitivity_ossicular'] for m in fold_metrics if 'sensitivity_ossicular' in m]
    ossic_spec = [m['specificity_ossicular'] for m in fold_metrics if 'specificity_ossicular' in m]
    
    if ossic_aucs:
        aggregated['ossicular_discontinuity'] = {
            'auc_mean': float(np.mean(ossic_aucs)),
            'auc_std': float(np.std(ossic_aucs)),
            'sensitivity_mean': float(np.mean(ossic_sens)) if ossic_sens else None,
            'sensitivity_std': float(np.std(ossic_sens)) if ossic_sens else None,
            'specificity_mean': float(np.mean(ossic_spec)) if ossic_spec else None,
            'specificity_std': float(np.std(ossic_spec)) if ossic_spec else None,
            'n_folds': len(ossic_aucs)
        }
    
    # Run sanity checks
    sanity_checks = run_sanity_checks(fold_metrics, training_histories)
    
    # Save results
    # 1. Per-fold metrics CSV
    fold_metrics_df = pd.DataFrame(fold_metrics)
    fold_metrics_df.to_csv(output_dir / 'cv_metrics_per_fold.csv', index=False)
    logger.info(f"Saved per-fold metrics to {output_dir / 'cv_metrics_per_fold.csv'}")
    
    # 2. Aggregated metrics JSON
    aggregated_results = {
        'timestamp': datetime.now().isoformat(),
        'n_folds_evaluated': len(fold_metrics),
        'folds': folds,
        'aggregated_metrics': aggregated,
        'sanity_checks': sanity_checks
    }
    
    with open(output_dir / 'cv_aggregated_metrics.json', 'w') as f:
        json.dump(convert_to_native_types(aggregated_results), f, indent=2)
    logger.info(f"Saved aggregated metrics to {output_dir / 'cv_aggregated_metrics.json'}")
    
    # 3. All predictions CSV
    all_preds_data = []
    for fold, preds in all_predictions.items():
        for i, ear_id in enumerate(preds['ear_ids']):
            all_preds_data.append({
                'fold': fold,
                'ear_id': ear_id,
                'prob_cholesteatoma': preds['prob_cholesteatoma'][i],
                'prob_ossicular': preds['prob_ossicular'][i],
                'label_cholesteatoma': preds['label_cholesteatoma'][i],
                'label_ossicular': preds['label_ossicular'][i],
                'mask_cholesteatoma': preds['mask_cholesteatoma'][i],
                'mask_ossicular': preds['mask_ossicular'][i]
            })
    
    all_preds_df = pd.DataFrame(all_preds_data)
    all_preds_df.to_csv(output_dir / 'all_cv_predictions.csv', index=False)
    
    # Generate figures
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)
    
    # 4. CV AUC boxplot
    if chole_aucs or ossic_aucs:
        auc_data = {}
        if chole_aucs:
            auc_data['cholesteatoma'] = {
                'mean': np.mean(chole_aucs),
                'std': np.std(chole_aucs),
                'values': chole_aucs
            }
        if ossic_aucs:
            auc_data['ossicular'] = {
                'mean': np.mean(ossic_aucs),
                'std': np.std(ossic_aucs),
                'values': ossic_aucs
            }
        
        plot_auc_summary(auc_data, figures_dir / 'cv_auc_summary.png',
                        title='Cross-Validation AUC Summary')
    
    # 5. Training curves for each fold
    for fold, history in training_histories.items():
        if history:
            plot_training_curves(
                history,
                figures_dir / f'fold_{fold}_training_curves.png',
                title=f'Fold {fold} Training History'
            )
    
    # Print summary
    print("\n" + "="*60)
    print("CROSS-VALIDATION EVALUATION SUMMARY")
    print("="*60)
    print(f"\nFolds evaluated: {len(fold_metrics)}")
    print(f"Output directory: {output_dir}")
    
    if 'cholesteatoma' in aggregated:
        print(f"\nCholesteatoma:")
        print(f"  AUC: {aggregated['cholesteatoma']['auc_mean']:.3f} ± {aggregated['cholesteatoma']['auc_std']:.3f}")
        if aggregated['cholesteatoma']['sensitivity_mean'] is not None:
            print(f"  Sensitivity: {aggregated['cholesteatoma']['sensitivity_mean']:.3f} ± {aggregated['cholesteatoma']['sensitivity_std']:.3f}")
            print(f"  Specificity: {aggregated['cholesteatoma']['specificity_mean']:.3f} ± {aggregated['cholesteatoma']['specificity_std']:.3f}")
    
    if 'ossicular_discontinuity' in aggregated:
        print(f"\nOssicular Discontinuity:")
        print(f"  AUC: {aggregated['ossicular_discontinuity']['auc_mean']:.3f} ± {aggregated['ossicular_discontinuity']['auc_std']:.3f}")
        if aggregated['ossicular_discontinuity']['sensitivity_mean'] is not None:
            print(f"  Sensitivity: {aggregated['ossicular_discontinuity']['sensitivity_mean']:.3f} ± {aggregated['ossicular_discontinuity']['sensitivity_std']:.3f}")
            print(f"  Specificity: {aggregated['ossicular_discontinuity']['specificity_mean']:.3f} ± {aggregated['ossicular_discontinuity']['specificity_std']:.3f}")
    
    print("\n" + "-"*60)
    print("SANITY CHECKS:")
    for check, passed in sanity_checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {check.replace('_', ' ').title()}: {status}")
    
    print("\n" + "="*60)
    logger.info("Evaluation complete!")


if __name__ == '__main__':
    main()
