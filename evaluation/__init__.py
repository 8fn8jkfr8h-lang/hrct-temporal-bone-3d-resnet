"""
Evaluation module for Temporal Bone HRCT project.

Provides metrics computation, visualization, and interpretability tools.
"""

from .metrics import (
    bootstrap_auc,
    bootstrap_metric,
    find_optimal_threshold,
    compute_classification_metrics,
    sanity_check_predictions
)

from .visualization import (
    plot_roc_curves,
    plot_pr_curves,
    plot_confusion_matrix,
    plot_cv_metrics,
    plot_training_curves
)

__all__ = [
    # Metrics
    'bootstrap_auc',
    'bootstrap_metric', 
    'find_optimal_threshold',
    'compute_classification_metrics',
    'sanity_check_predictions',
    # Visualization
    'plot_roc_curves',
    'plot_pr_curves',
    'plot_confusion_matrix',
    'plot_cv_metrics',
    'plot_training_curves',
]
