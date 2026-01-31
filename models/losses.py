"""
Loss functions for multi-task temporal bone pathology detection.

Implements masked BCE loss that handles missing labels (NULL values)
with class weighting for imbalanced datasets.
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional


class MaskedMultiTaskLoss(nn.Module):
    """
    Masked multi-task BCE loss for pathology detection.
    
    Handles:
    - Multiple binary classification tasks
    - Missing labels (masked out during loss computation)
    - Class imbalance via positive weights
    - Task-specific weighting
    """
    
    def __init__(
        self,
        pos_weights: Optional[Dict[str, float]] = None,
        task_weights: Optional[Dict[str, float]] = None,
        num_tasks: int = 2
    ):
        """
        Args:
            pos_weights: Positive class weights per task (neg_count / pos_count)
                        Keys: 'chole', 'ossic', 'facial'
            task_weights: Task importance weights (should sum to 1)
                         Keys: 'chole', 'ossic', 'facial'
            num_tasks: Number of classification tasks (2 or 3)
        """
        super().__init__()
        
        self.num_tasks = num_tasks
        
        # Default weights if not provided
        if pos_weights is None:
            pos_weights = {'chole': 1.0, 'ossic': 1.0, 'facial': 1.0}
        
        if task_weights is None:
            if num_tasks == 2:
                task_weights = {'chole': 0.6, 'ossic': 0.4}
            else:
                task_weights = {'chole': 0.5, 'ossic': 0.3, 'facial': 0.2}
        
        self.task_weights = task_weights
        
        # Create loss functions with positive weights
        self.loss_chole = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weights['chole']]),
            reduction='none'
        )
        self.loss_ossic = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weights['ossic']]),
            reduction='none'
        )
        
        if num_tasks >= 3:
            self.loss_facial = nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor([pos_weights.get('facial', 1.0)]),
                reduction='none'
            )
        else:
            self.loss_facial = None
    
    def forward(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        masks: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute masked multi-task loss.
        
        Args:
            preds: Predicted logits (B, num_tasks)
            targets: Ground truth labels (B, num_tasks)
            masks: Valid label mask (B, num_tasks) - 1 if valid, 0 if NULL
            
        Returns:
            Tuple of:
            - Total weighted loss (scalar)
            - Dictionary with per-task losses
        """
        device = preds.device
        
        # Move pos_weights to correct device
        self.loss_chole.pos_weight = self.loss_chole.pos_weight.to(device)
        self.loss_ossic.pos_weight = self.loss_ossic.pos_weight.to(device)
        if self.loss_facial is not None:
            self.loss_facial.pos_weight = self.loss_facial.pos_weight.to(device)
        
        # Compute per-sample losses
        loss_c = self.loss_chole(preds[:, 0], targets[:, 0].float())
        loss_o = self.loss_ossic(preds[:, 1], targets[:, 1].float())
        
        # Apply masks
        loss_c = loss_c * masks[:, 0]
        loss_o = loss_o * masks[:, 1]
        
        # Normalize by number of valid samples
        n_valid_c = masks[:, 0].sum().clamp(min=1)
        n_valid_o = masks[:, 1].sum().clamp(min=1)
        
        loss_c_mean = loss_c.sum() / n_valid_c
        loss_o_mean = loss_o.sum() / n_valid_o
        
        # Weighted sum
        total_loss = (
            self.task_weights['chole'] * loss_c_mean +
            self.task_weights['ossic'] * loss_o_mean
        )
        
        loss_dict = {
            'loss_chole': loss_c_mean.item(),
            'loss_ossic': loss_o_mean.item()
        }
        
        # Add facial nerve loss if applicable
        if self.num_tasks >= 3 and self.loss_facial is not None:
            loss_f = self.loss_facial(preds[:, 2], targets[:, 2].float())
            loss_f = loss_f * masks[:, 2]
            n_valid_f = masks[:, 2].sum().clamp(min=1)
            loss_f_mean = loss_f.sum() / n_valid_f
            
            total_loss = total_loss + self.task_weights['facial'] * loss_f_mean
            loss_dict['loss_facial'] = loss_f_mean.item()
        
        loss_dict['loss_total'] = total_loss.item()
        
        return total_loss, loss_dict


def compute_class_weights(labels_df, label_cols: list) -> Dict[str, float]:
    """
    Compute positive class weights from label distribution.
    
    Args:
        labels_df: DataFrame with labels
        label_cols: List of label column names
        
    Returns:
        Dictionary of positive weights (neg_count / pos_count)
    """
    weights = {}
    
    col_mapping = {
        'cholesteatoma': 'chole',
        'ossicular_discontinuity': 'ossic',
        'facial_dehiscence': 'facial'
    }
    
    for col in label_cols:
        if col in labels_df.columns:
            valid = labels_df[col].dropna()
            n_pos = (valid == 1).sum()
            n_neg = (valid == 0).sum()
            
            if n_pos > 0:
                weight = n_neg / n_pos
            else:
                weight = 1.0
            
            key = col_mapping.get(col, col)
            weights[key] = weight
    
    return weights


if __name__ == "__main__":
    # Test the loss function
    print("Testing MaskedMultiTaskLoss...")
    
    # Create dummy data
    batch_size = 8
    num_tasks = 2
    
    preds = torch.randn(batch_size, num_tasks)
    targets = torch.randint(0, 2, (batch_size, num_tasks)).float()
    masks = torch.ones(batch_size, num_tasks)
    masks[0, 1] = 0  # Simulate a missing label
    
    # Create loss function
    criterion = MaskedMultiTaskLoss(
        pos_weights={'chole': 0.5, 'ossic': 1.2},
        num_tasks=num_tasks
    )
    
    loss, loss_dict = criterion(preds, targets, masks)
    
    print(f"Total loss: {loss.item():.4f}")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.4f}")
