"""
Phase 5: Model Evaluation (Production Script)
Temporal Bone HRCT Project

Comprehensive evaluation script for full production dataset:
- Test set evaluation with ensemble predictions from 5 folds
- Per-pathology metrics for cholesteatoma, ossicular, facial nerve
- Bootstrap 95% CIs for all metrics
- Error analysis (FP/FN case identification)
- Grad-CAM visualization generation
- ROC curves, PR curves, confusion matrices

Usage:
    python pipeline/phase5_model_evaluation.py \
        --models_dir models \
        --split_dir dataset_splits \
        --roi_dir roi_extracted \
        --labels_csv labels.csv \
        --output_dir evaluation_results
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.resnet3d import TemporalBoneClassifier
from data.dataset import TemporalBoneDataset
from data.transforms import get_val_transforms
from evaluation.metrics import (
    bootstrap_auc,
    bootstrap_metric,
    compute_classification_metrics,
    find_optimal_threshold,
    sanity_check_predictions
)
from evaluation.visualization import (
    plot_roc_curves,
    plot_pr_curves,
    plot_confusion_matrix
)
from evaluation.gradcam import generate_gradcam_for_batch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Task configuration
TASK_NAMES = ['cholesteatoma', 'ossicular_discontinuity', 'facial_nerve_dehiscence']
TASK_WEIGHTS = {'cholesteatoma': 0.5, 'ossicular': 0.3, 'facial_nerve': 0.2}


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


def load_ensemble_models(
    models_dir: Path,
    device: torch.device,
    num_tasks: int = 3,
    n_folds: int = 5
) -> List[TemporalBoneClassifier]:
    """
    Load all fold models for ensemble prediction.
    
    Args:
        models_dir: Directory containing fold subdirectories
        device: Device to load models on
        num_tasks: Number of classification tasks
        n_folds: Number of folds
        
    Returns:
        List of loaded models
    """
    models = []
    
    for fold in range(n_folds):
        fold_dir = models_dir / f'fold_{fold}'
        checkpoint_path = fold_dir / 'best_checkpoint.pth'
        
        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found for fold {fold}: {checkpoint_path}")
            continue
        
        try:
            # Load config if available
            config_path = fold_dir / 'config.json'
            use_cbam = True
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                    use_cbam = config.get('use_cbam', True)
            
            # Create and load model
            model = TemporalBoneClassifier(
                num_tasks=num_tasks,
                use_cbam=use_cbam,
                pretrained_path=None
            )
            
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.to(device)
            model.eval()
            models.append(model)
            
            logger.info(f"Loaded model from fold {fold}")
            
        except Exception as e:
            logger.error(f"Failed to load model from fold {fold}: {e}")
    
    return models


def ensemble_predict(
    models: List[nn.Module],
    dataloader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Generate ensemble predictions from multiple models.
    
    Args:
        models: List of trained models
        dataloader: DataLoader for test data
        device: Device for inference
        
    Returns:
        Tuple of (ensemble_probs, labels, masks, ear_ids)
    """
    all_probs = []
    all_labels = []
    all_masks = []
    all_ear_ids = []
    
    # First pass: collect all model predictions
    model_predictions = {i: [] for i in range(len(models))}
    
    for model_idx, model in enumerate(models):
        model.eval()
        
        with torch.no_grad():
            for batch in dataloader:
                volumes = batch['image'].to(device)
                
                outputs = model(volumes)
                probs = torch.sigmoid(outputs).cpu().numpy()
                model_predictions[model_idx].append(probs)
                
                # Only collect labels once (first model)
                if model_idx == 0:
                    all_labels.append(batch['labels'].numpy())
                    all_masks.append(batch['mask'].numpy())
                    all_ear_ids.extend(batch['ear_id'])
    
    # Concatenate predictions
    for model_idx in model_predictions:
        model_predictions[model_idx] = np.concatenate(model_predictions[model_idx], axis=0)
    
    all_labels = np.concatenate(all_labels, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)
    
    # Ensemble: average probabilities across models
    stacked_preds = np.stack([model_predictions[i] for i in range(len(models))], axis=0)
    ensemble_probs = np.mean(stacked_preds, axis=0)
    
    return ensemble_probs, all_labels, all_masks, all_ear_ids


def evaluate_pathology(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
    pathology_name: str,
    n_bootstrap: int = 1000
) -> Dict:
    """
    Compute comprehensive metrics for a single pathology.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        mask: Valid sample mask
        pathology_name: Name of pathology for logging
        n_bootstrap: Number of bootstrap iterations for CIs
        
    Returns:
        Dictionary with all metrics
    """
    valid = mask == 1
    n_valid = valid.sum()
    
    if n_valid < 10:
        logger.warning(f"Insufficient samples for {pathology_name}: {n_valid}")
        return {'n_valid': int(n_valid), 'status': 'insufficient_data'}
    
    y_true_valid = y_true[valid]
    y_pred_valid = y_pred[valid]
    
    # Sanity check
    sanity_check_predictions(y_pred_valid, f"{pathology_name} predictions")
    
    # AUC with bootstrap CI
    auc, auc_ci_lower, auc_ci_upper = bootstrap_auc(
        y_true_valid, y_pred_valid, n_bootstrap=n_bootstrap
    )
    
    # Optimal thresholds
    threshold_f1, f1_score = find_optimal_threshold(y_true_valid, y_pred_valid, metric='f1')
    threshold_youden, youden_j = find_optimal_threshold(y_true_valid, y_pred_valid, metric='youden')
    
    # Classification metrics at F1-optimal threshold
    metrics = compute_classification_metrics(y_true_valid, y_pred_valid, threshold_f1)
    
    # Bootstrap CIs for sensitivity and specificity
    def sensitivity_fn(y_t, y_p):
        y_p_bin = (y_p >= threshold_f1).astype(int)
        if len(np.unique(y_t)) < 2:
            return 0.5
        cm = np.bincount(y_t * 2 + y_p_bin, minlength=4)
        tn, fp, fn, tp = cm[0], cm[1], cm[2], cm[3]
        return tp / (tp + fn) if (tp + fn) > 0 else 0
    
    def specificity_fn(y_t, y_p):
        y_p_bin = (y_p >= threshold_f1).astype(int)
        if len(np.unique(y_t)) < 2:
            return 0.5
        cm = np.bincount(y_t * 2 + y_p_bin, minlength=4)
        tn, fp, fn, tp = cm[0], cm[1], cm[2], cm[3]
        return tn / (tn + fp) if (tn + fp) > 0 else 0
    
    _, sens_ci_lower, sens_ci_upper = bootstrap_metric(
        y_true_valid, y_pred_valid, sensitivity_fn, n_bootstrap=500
    )
    _, spec_ci_lower, spec_ci_upper = bootstrap_metric(
        y_true_valid, y_pred_valid, specificity_fn, n_bootstrap=500
    )
    
    results = {
        'n_valid': int(n_valid),
        'n_positive': int(y_true_valid.sum()),
        'n_negative': int((1 - y_true_valid).sum()),
        'auc': float(auc),
        'auc_ci': [float(auc_ci_lower), float(auc_ci_upper)],
        'threshold_f1': float(threshold_f1),
        'threshold_youden': float(threshold_youden),
        'sensitivity': float(metrics['sensitivity']),
        'sensitivity_ci': [float(sens_ci_lower), float(sens_ci_upper)],
        'specificity': float(metrics['specificity']),
        'specificity_ci': [float(spec_ci_lower), float(spec_ci_upper)],
        'ppv': float(metrics['ppv']),
        'npv': float(metrics['npv']),
        'f1': float(metrics['f1']),
        'accuracy': float(metrics['accuracy']),
        'confusion_matrix': metrics['confusion_matrix'],
        'y_true': y_true_valid.tolist(),
        'y_pred': y_pred_valid.tolist(),
        'status': 'success'
    }
    
    return results


def identify_error_cases(
    ear_ids: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mask: np.ndarray,
    threshold: float,
    pathology_name: str,
    output_dir: Path
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identify and save false positive/negative cases for review.
    
    Args:
        ear_ids: List of ear identifiers
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        mask: Valid sample mask
        threshold: Classification threshold
        pathology_name: Name of pathology
        output_dir: Directory to save error CSVs
        
    Returns:
        Tuple of (false_positives_df, false_negatives_df)
    """
    valid = mask == 1
    
    # Create DataFrame
    df = pd.DataFrame({
        'ear_id': ear_ids,
        'true_label': y_true,
        'pred_prob': y_pred,
        'pred_label': (y_pred >= threshold).astype(int),
        'valid': valid
    })
    
    # Filter to valid samples
    df_valid = df[df['valid']].copy()
    
    # Identify errors
    fp_df = df_valid[(df_valid['true_label'] == 0) & (df_valid['pred_label'] == 1)]
    fn_df = df_valid[(df_valid['true_label'] == 1) & (df_valid['pred_label'] == 0)]
    
    # Save
    fp_path = output_dir / f'false_positives_{pathology_name}.csv'
    fn_path = output_dir / f'false_negatives_{pathology_name}.csv'
    
    fp_df.to_csv(fp_path, index=False)
    fn_df.to_csv(fn_path, index=False)
    
    logger.info(f"{pathology_name}: {len(fp_df)} false positives, {len(fn_df)} false negatives")
    
    return fp_df, fn_df


def main():
    parser = argparse.ArgumentParser(
        description='Phase 5: Model Evaluation (Production)'
    )
    parser.add_argument('--models_dir', type=str, default='models',
                       help='Directory with trained models')
    parser.add_argument('--split_dir', type=str, default='dataset_splits',
                       help='Directory with dataset splits')
    parser.add_argument('--roi_dir', type=str, default='roi_extracted',
                       help='Directory with extracted ROIs')
    parser.add_argument('--labels_csv', type=str, default='labels.csv',
                       help='Path to labels CSV')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Output directory for evaluation results')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for inference')
    parser.add_argument('--n_bootstrap', type=int, default=1000,
                       help='Number of bootstrap iterations for CIs')
    parser.add_argument('--generate_gradcam', action='store_true',
                       help='Generate Grad-CAM visualizations')
    parser.add_argument('--gradcam_samples', type=int, default=10,
                       help='Number of samples for Grad-CAM')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device (auto, cpu, cuda)')
    parser.add_argument('--num_tasks', type=int, default=3,
                       help='Number of classification tasks (2 for validation, 3 for production)')
    
    args = parser.parse_args()
    
    # Validate inputs
    validate_inputs(args)
    
    # Setup paths
    models_dir = Path(args.models_dir)
    split_dir = Path(args.split_dir)
    roi_dir = Path(args.roi_dir)
    output_dir = Path(args.output_dir)
    
    test_output = output_dir / 'test_set'
    cv_output = output_dir / 'cross_validation'
    figures_output = test_output / 'figures'
    gradcam_output = test_output / 'gradcam_cases'
    
    for d in [test_output, cv_output, figures_output, gradcam_output]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load labels
    labels_df = pd.read_csv(args.labels_csv)
    logger.info(f"Loaded {len(labels_df)} label entries")
    
    # Load test set
    test_set_path = split_dir / 'test_set.json'
    if not test_set_path.exists():
        logger.error(f"Test set not found: {test_set_path}")
        logger.info("Falling back to cross-validation evaluation only")
        # Could add CV-only evaluation here
        return
    
    with open(test_set_path) as f:
        test_set = json.load(f)
    
    test_ear_ids = test_set['ear_ids']
    logger.info(f"Test set: {len(test_ear_ids)} ears")
    
    # Load ensemble models
    models = load_ensemble_models(models_dir, device, num_tasks=args.num_tasks)
    
    if not models:
        logger.error("No models could be loaded")
        return
    
    logger.info(f"Loaded {len(models)} models for ensemble")
    
    # Create test dataset and dataloader
    task_names = TASK_NAMES[:args.num_tasks]
    test_dataset = TemporalBoneDataset(
        ear_ids=test_ear_ids,
        roi_dir=roi_dir,
        labels_df=labels_df,
        transforms=get_val_transforms(),
        num_tasks=args.num_tasks
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    
    # Generate ensemble predictions
    logger.info("Generating ensemble predictions on test set...")
    ensemble_probs, labels, masks, ear_ids = ensemble_predict(
        models, test_loader, device
    )
    
    # Evaluate each pathology
    all_results = {}
    roc_data = {}
    
    for task_idx, task_name in enumerate(task_names):
        logger.info(f"Evaluating {task_name}...")
        
        results = evaluate_pathology(
            y_true=labels[:, task_idx],
            y_pred=ensemble_probs[:, task_idx],
            mask=masks[:, task_idx],
            pathology_name=task_name,
            n_bootstrap=args.n_bootstrap
        )
        
        all_results[task_name] = results
        
        if results['status'] == 'success':
            # Store data for ROC plotting
            roc_data[task_name] = {
                'y_true': np.array(results['y_true']),
                'y_pred': np.array(results['y_pred']),
                'auc': results['auc'],
                'auc_ci': tuple(results['auc_ci'])
            }
            
            # Identify error cases
            identify_error_cases(
                ear_ids=ear_ids,
                y_true=labels[:, task_idx],
                y_pred=ensemble_probs[:, task_idx],
                mask=masks[:, task_idx],
                threshold=results['threshold_f1'],
                pathology_name=task_name,
                output_dir=test_output
            )
            
            # Plot confusion matrix
            y_pred_binary = (np.array(results['y_pred']) >= results['threshold_f1']).astype(int)
            plot_confusion_matrix(
                y_true=np.array(results['y_true']),
                y_pred=y_pred_binary,
                output_path=figures_output / f'confusion_matrix_{task_name}.png',
                title=f'Confusion Matrix - {task_name.replace("_", " ").title()}'
            )
    
    # Plot combined ROC curves
    if roc_data:
        plot_roc_curves(
            roc_data,
            figures_output / 'roc_curves_all_pathologies.png',
            title='ROC Curves - Test Set Evaluation'
        )
        
        plot_pr_curves(
            roc_data,
            figures_output / 'pr_curves_all_pathologies.png',
            title='Precision-Recall Curves - Test Set Evaluation'
        )
    
    # Save ensemble predictions
    predictions_df = pd.DataFrame({
        'ear_id': ear_ids
    })
    for task_idx, task_name in enumerate(task_names):
        predictions_df[f'true_{task_name}'] = labels[:, task_idx]
        predictions_df[f'pred_{task_name}'] = (ensemble_probs[:, task_idx] >= all_results[task_name].get('threshold_f1', 0.5)).astype(int)
        predictions_df[f'prob_{task_name}'] = ensemble_probs[:, task_idx]
    
    predictions_df.to_csv(test_output / 'predictions_ensemble.csv', index=False)
    
    # Save metrics summary (without large arrays)
    metrics_summary = {}
    for task_name, results in all_results.items():
        metrics_summary[task_name] = {k: v for k, v in results.items() 
                                      if k not in ['y_true', 'y_pred']}
    
    with open(test_output / 'metrics_summary.json', 'w') as f:
        json.dump(metrics_summary, f, indent=2)
    
    # Generate Grad-CAM if requested
    if args.generate_gradcam and models:
        logger.info("Generating Grad-CAM visualizations...")
        try:
            generate_gradcam_for_batch(
                model=models[0],  # Use first model for Grad-CAM
                dataloader=test_loader,
                device=device,
                output_dir=gradcam_output,
                max_samples=args.gradcam_samples,
                task_idx=0  # Cholesteatoma
            )
        except Exception as e:
            logger.error(f"Grad-CAM generation failed: {e}")
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SET EVALUATION SUMMARY")
    print("="*70)
    print(f"\nTest samples: {len(ear_ids)}")
    print(f"Models in ensemble: {len(models)}")
    print(f"Output directory: {output_dir}")
    
    for task_name, results in all_results.items():
        print(f"\n{task_name.replace('_', ' ').title()}:")
        if results['status'] == 'success':
            print(f"  AUC: {results['auc']:.3f} (95% CI: [{results['auc_ci'][0]:.3f}, {results['auc_ci'][1]:.3f}])")
            print(f"  Sensitivity: {results['sensitivity']:.3f} (95% CI: [{results['sensitivity_ci'][0]:.3f}, {results['sensitivity_ci'][1]:.3f}])")
            print(f"  Specificity: {results['specificity']:.3f} (95% CI: [{results['specificity_ci'][0]:.3f}, {results['specificity_ci'][1]:.3f}])")
            print(f"  PPV: {results['ppv']:.3f}, NPV: {results['npv']:.3f}")
            print(f"  F1 Score: {results['f1']:.3f}")
            print(f"  Threshold (F1): {results['threshold_f1']:.3f}")
        else:
            print(f"  Status: {results['status']}")
    
    print("\n" + "="*70)
    logger.info("Evaluation complete!")


if __name__ == '__main__':
    main()
