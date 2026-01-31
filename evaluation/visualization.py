"""
Visualization functions for model evaluation.

Provides plotting utilities for ROC curves, confusion matrices,
training history, and cross-validation results.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
import logging

logger = logging.getLogger(__name__)

# Default figure style
plt.style.use('seaborn-v0_8-whitegrid')
FIGSIZE_SINGLE = (8, 6)
FIGSIZE_DOUBLE = (12, 5)
FIGSIZE_QUAD = (14, 10)
DPI = 150


def plot_roc_curves(
    results: Dict[str, Dict],
    output_path: Path,
    title: str = "ROC Curves",
    show_ci: bool = True
) -> None:
    """
    Plot ROC curves for multiple pathologies.
    
    Args:
        results: Dict with keys as pathology names, values containing:
                 - 'y_true': ground truth labels
                 - 'y_pred': predicted probabilities
                 - 'auc': AUC score
                 - 'auc_ci' (optional): tuple of (ci_lower, ci_upper)
        output_path: Path to save the figure
        title: Figure title
        show_ci: Whether to show confidence intervals in legend
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    
    colors = ['#2E86AB', '#E94F37', '#44AF69', '#F18F01']
    
    for idx, (pathology, data) in enumerate(results.items()):
        if 'y_true' not in data or 'y_pred' not in data:
            continue
            
        fpr, tpr, _ = roc_curve(data['y_true'], data['y_pred'])
        
        auc = data.get('auc', 0.5)
        if show_ci and 'auc_ci' in data:
            label = f"{pathology} (AUC = {auc:.3f}, 95% CI: [{data['auc_ci'][0]:.3f}, {data['auc_ci'][1]:.3f}])"
        else:
            label = f"{pathology} (AUC = {auc:.3f})"
        
        color = colors[idx % len(colors)]
        ax.plot(fpr, tpr, color=color, linewidth=2, label=label)
    
    # Diagonal reference line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random Classifier')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved ROC curves to {output_path}")


def plot_pr_curves(
    results: Dict[str, Dict],
    output_path: Path,
    title: str = "Precision-Recall Curves"
) -> None:
    """
    Plot precision-recall curves for multiple pathologies.
    
    Args:
        results: Dict with pathology names as keys, values containing y_true, y_pred
        output_path: Path to save the figure
        title: Figure title
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    
    colors = ['#2E86AB', '#E94F37', '#44AF69', '#F18F01']
    
    for idx, (pathology, data) in enumerate(results.items()):
        if 'y_true' not in data or 'y_pred' not in data:
            continue
            
        precision, recall, _ = precision_recall_curve(data['y_true'], data['y_pred'])
        
        # Calculate average precision (area under PR curve)
        from sklearn.metrics import average_precision_score
        ap = average_precision_score(data['y_true'], data['y_pred'])
        
        color = colors[idx % len(colors)]
        ax.plot(recall, precision, color=color, linewidth=2, 
                label=f"{pathology} (AP = {ap:.3f})")
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall (Sensitivity)', fontsize=12)
    ax.set_ylabel('Precision (PPV)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved PR curves to {output_path}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    title: str = "Confusion Matrix",
    class_names: Tuple[str, str] = ('Negative', 'Positive')
) -> None:
    """
    Plot confusion matrix as a heatmap.
    
    Args:
        y_true: Ground truth binary labels
        y_pred: Predicted binary labels
        output_path: Path to save the figure
        title: Figure title
        class_names: Names for negative and positive classes
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Create heatmap
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Count', rotation=-90, va="bottom")
    
    # Set labels
    ax.set(xticks=[0, 1],
           yticks=[0, 1],
           xticklabels=[f'Pred: {class_names[0]}', f'Pred: {class_names[1]}'],
           yticklabels=[f'True: {class_names[0]}', f'True: {class_names[1]}'])
    
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    total = cm.sum()
    thresh = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct = 100 * count / total
            text_color = "white" if cm[i, j] > thresh else "black"
            ax.text(j, i, f"{count}\n({pct:.1f}%)",
                   ha="center", va="center", color=text_color, fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved confusion matrix to {output_path}")


def plot_cv_metrics(
    fold_metrics: List[Dict[str, float]],
    output_path: Path,
    metric_names: Optional[List[str]] = None,
    title: str = "Cross-Validation Metrics"
) -> None:
    """
    Plot cross-validation metrics as boxplots.
    
    Args:
        fold_metrics: List of metric dictionaries, one per fold
        output_path: Path to save the figure
        metric_names: List of metric names to plot (default: auto-detect)
        title: Figure title
    """
    if not fold_metrics:
        logger.warning("No fold metrics to plot")
        return
    
    # Auto-detect metric names if not provided
    if metric_names is None:
        exclude_keys = ['confusion_matrix', 'fold', 'ear_id', 'patient_id']
        metric_names = [k for k in fold_metrics[0].keys() 
                       if isinstance(fold_metrics[0][k], (int, float)) 
                       and k not in exclude_keys
                       and not np.isnan(fold_metrics[0][k])]
    
    # Prepare data for boxplot
    data = []
    labels = []
    for metric in metric_names:
        values = [fm[metric] for fm in fold_metrics if metric in fm and not np.isnan(fm[metric])]
        if values:
            data.append(values)
            labels.append(metric.replace('_', ' ').title())
    
    if not data:
        logger.warning("No valid metrics to plot")
        return
    
    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    
    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    
    # Color the boxes
    colors = ['#2E86AB', '#E94F37', '#44AF69', '#F18F01', '#8338EC', '#FF006E']
    for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.05])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Rotate x labels if many
    if len(labels) > 4:
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved CV metrics boxplot to {output_path}")


def plot_training_curves(
    history: Dict[str, List[float]],
    output_path: Path,
    title: str = "Training History"
) -> None:
    """
    Plot training and validation loss/metrics over epochs.
    
    Args:
        history: Dictionary with keys like 'train_loss', 'val_loss', 'val_auc_chole', etc.
        output_path: Path to save the figure
        title: Figure title
    """
    # Determine what to plot
    has_loss = 'train_loss' in history or 'val_loss' in history
    auc_keys = [k for k in history.keys() if 'auc' in k.lower()]
    
    n_plots = int(has_loss) + int(len(auc_keys) > 0)
    if n_plots == 0:
        logger.warning("No training history to plot")
        return
    
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]
    
    idx = 0
    
    # Plot loss
    if has_loss:
        ax = axes[idx]
        epochs = range(1, len(history.get('train_loss', history.get('val_loss', []))) + 1)
        
        if 'train_loss' in history:
            ax.plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Training Loss')
        if 'val_loss' in history:
            ax.plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Validation Loss')
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training & Validation Loss', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        idx += 1
    
    # Plot AUC metrics
    if auc_keys:
        ax = axes[idx]
        epochs = range(1, len(history[auc_keys[0]]) + 1)
        
        colors = ['#2E86AB', '#E94F37', '#44AF69', '#F18F01']
        for i, key in enumerate(auc_keys):
            label = key.replace('val_auc_', '').replace('_', ' ').title()
            ax.plot(epochs, history[key], color=colors[i % len(colors)], 
                   linewidth=2, label=label)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('AUC-ROC', fontsize=12)
        ax.set_title('Validation AUC', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right')
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved training curves to {output_path}")


def plot_auc_summary(
    aggregated_metrics: Dict[str, Dict],
    output_path: Path,
    title: str = "Cross-Validation AUC Summary"
) -> None:
    """
    Plot aggregated AUC with error bars for each pathology.
    
    Args:
        aggregated_metrics: Dict with pathology names, each containing 'mean', 'std', 'values'
        output_path: Path to save figure
        title: Figure title
    """
    pathologies = list(aggregated_metrics.keys())
    means = [aggregated_metrics[p]['mean'] for p in pathologies]
    stds = [aggregated_metrics[p]['std'] for p in pathologies]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    colors = ['#2E86AB', '#E94F37', '#44AF69']
    x = np.arange(len(pathologies))
    
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors[:len(pathologies)], 
                  alpha=0.7, edgecolor='black')
    
    # Add value labels
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + std + 0.02,
               f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontsize=10)
    
    ax.set_xticks(x)
    ax.set_xticklabels([p.replace('_', ' ').title() for p in pathologies])
    ax.set_ylabel('AUC-ROC', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1.15])
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random (0.5)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=DPI, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"Saved AUC summary to {output_path}")
