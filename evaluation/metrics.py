"""
Metrics computation for model evaluation.

Provides functions for computing classification metrics with bootstrap
confidence intervals, optimal threshold selection, and prediction validation.
"""

import numpy as np
from typing import Dict, Tuple, Callable
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    confusion_matrix, f1_score
)
import logging

logger = logging.getLogger(__name__)


def bootstrap_auc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 42
) -> Tuple[float, float, float]:
    """
    Compute AUC-ROC with bootstrap confidence intervals.
    
    Args:
        y_true: Ground truth binary labels (N,)
        y_pred: Predicted probabilities (N,)
        n_bootstrap: Number of bootstrap iterations
        confidence_level: Confidence level for CI (default 0.95 for 95% CI)
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (auc, ci_lower, ci_upper)
    """
    np.random.seed(random_seed)
    n_samples = len(y_true)
    
    # Check for valid data
    if len(np.unique(y_true)) < 2:
        logger.warning("Only one class present in y_true. AUC is undefined.")
        return 0.5, 0.5, 0.5
    
    # Compute base AUC
    base_auc = roc_auc_score(y_true, y_pred)
    
    # Bootstrap
    bootstrap_aucs = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        
        # Skip if only one class in bootstrap sample
        if len(np.unique(y_true_boot)) < 2:
            continue
            
        try:
            auc = roc_auc_score(y_true_boot, y_pred_boot)
            bootstrap_aucs.append(auc)
        except ValueError:
            continue
    
    if len(bootstrap_aucs) < n_bootstrap * 0.5:
        logger.warning(f"Only {len(bootstrap_aucs)}/{n_bootstrap} valid bootstrap samples")
        return base_auc, base_auc, base_auc
    
    # Compute confidence intervals
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_aucs, alpha / 2 * 100)
    ci_upper = np.percentile(bootstrap_aucs, (1 - alpha / 2) * 100)
    
    return base_auc, ci_lower, ci_upper


def bootstrap_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_seed: int = 42
) -> Tuple[float, float, float]:
    """
    Compute arbitrary metric with bootstrap confidence intervals.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels or probabilities
        metric_fn: Function that computes the metric
        n_bootstrap: Number of bootstrap iterations
        confidence_level: Confidence level for CI
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (metric_value, ci_lower, ci_upper)
    """
    np.random.seed(random_seed)
    n_samples = len(y_true)
    
    # Compute base metric
    base_metric = metric_fn(y_true, y_pred)
    
    # Bootstrap
    bootstrap_metrics = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        
        try:
            metric = metric_fn(y_true_boot, y_pred_boot)
            bootstrap_metrics.append(metric)
        except Exception:
            continue
    
    if len(bootstrap_metrics) < n_bootstrap * 0.5:
        logger.warning(f"Only {len(bootstrap_metrics)}/{n_bootstrap} valid bootstrap samples")
        return base_metric, base_metric, base_metric
    
    # Compute confidence intervals
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_metrics, alpha / 2 * 100)
    ci_upper = np.percentile(bootstrap_metrics, (1 - alpha / 2) * 100)
    
    return base_metric, ci_lower, ci_upper


def find_optimal_threshold(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str = 'f1'
) -> Tuple[float, float]:
    """
    Find optimal classification threshold.
    
    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted probabilities
        metric: Optimization metric ('f1' or 'youden')
        
    Returns:
        Tuple of (optimal_threshold, metric_value_at_threshold)
    """
    if metric == 'youden':
        # Maximize Youden's J statistic (sensitivity + specificity - 1)
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        j_scores = tpr - fpr  # Youden's J = sensitivity + specificity - 1 = tpr - fpr
        best_idx = np.argmax(j_scores)
        return thresholds[best_idx], j_scores[best_idx]
    
    elif metric == 'f1':
        # Maximize F1 score
        thresholds = np.linspace(0.01, 0.99, 99)
        best_threshold = 0.5
        best_f1 = 0
        
        for thresh in thresholds:
            y_pred_binary = (y_pred >= thresh).astype(int)
            
            # Skip if all predictions are same class
            if len(np.unique(y_pred_binary)) < 2:
                continue
                
            f1 = f1_score(y_true, y_pred_binary, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = thresh
        
        return best_threshold, best_f1
    
    else:
        raise ValueError(f"Unknown metric: {metric}. Use 'f1' or 'youden'.")


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute comprehensive classification metrics.
    
    Args:
        y_true: Ground truth binary labels
        y_pred_prob: Predicted probabilities
        threshold: Classification threshold
        
    Returns:
        Dictionary with sensitivity, specificity, PPV, NPV, F1, accuracy
    """
    y_pred = (y_pred_prob >= threshold).astype(int)
    
    # Handle edge cases
    if len(np.unique(y_true)) < 2:
        logger.warning("Only one class in y_true. Metrics may be undefined.")
        return {
            'sensitivity': np.nan,
            'specificity': np.nan,
            'ppv': np.nan,
            'npv': np.nan,
            'f1': np.nan,
            'accuracy': np.nan
        }
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Handle case where predictions are all same class
    if cm.shape != (2, 2):
        logger.warning("Confusion matrix is not 2x2. Metrics may be undefined.")
        return {
            'sensitivity': np.nan,
            'specificity': np.nan,
            'ppv': np.nan,
            'npv': np.nan,
            'f1': np.nan,
            'accuracy': np.mean(y_true == y_pred)
        }
    
    tn, fp, fn, tp = cm.ravel()
    
    # Compute metrics with safe division
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    return {
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
        'f1': f1,
        'accuracy': accuracy,
        'confusion_matrix': [[int(tn), int(fp)], [int(fn), int(tp)]]
    }


def sanity_check_predictions(y_pred: np.ndarray, name: str = "predictions") -> bool:
    """
    Verify predictions are valid (no NaN/Inf, in [0,1] range for probabilities).
    
    Args:
        y_pred: Predicted values to check
        name: Name for logging purposes
        
    Returns:
        True if predictions are valid, raises ValueError otherwise
    """
    if np.any(np.isnan(y_pred)):
        raise ValueError(f"NaN values found in {name}")
    
    if np.any(np.isinf(y_pred)):
        raise ValueError(f"Inf values found in {name}")
    
    if np.any((y_pred < 0) | (y_pred > 1)):
        logger.warning(f"{name} contains values outside [0, 1] range. "
                      "Assuming these are logits, not probabilities.")
    
    logger.info(f"Sanity check passed for {name}: {len(y_pred)} samples, "
               f"range [{y_pred.min():.4f}, {y_pred.max():.4f}]")
    return True


def compute_auc_safe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute AUC-ROC with error handling for edge cases.
    
    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted probabilities
        
    Returns:
        AUC score or 0.5 if undefined
    """
    if len(np.unique(y_true)) < 2:
        return 0.5
    
    try:
        return roc_auc_score(y_true, y_pred)
    except ValueError as e:
        logger.warning(f"AUC computation failed: {e}")
        return 0.5


def aggregate_cv_metrics(fold_metrics: list) -> Dict[str, Dict[str, float]]:
    """
    Aggregate metrics across cross-validation folds.
    
    Args:
        fold_metrics: List of metric dictionaries, one per fold
        
    Returns:
        Dictionary with mean and std for each metric
    """
    # Get all metric keys from first fold
    if not fold_metrics:
        return {}
    
    metric_keys = [k for k in fold_metrics[0].keys() 
                   if isinstance(fold_metrics[0][k], (int, float)) and not np.isnan(fold_metrics[0][k])]
    
    aggregated = {}
    for key in metric_keys:
        values = [fm[key] for fm in fold_metrics if key in fm and not np.isnan(fm[key])]
        if values:
            aggregated[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'values': [float(v) for v in values]
            }
    
    return aggregated
