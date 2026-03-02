"""
Phase 4: Model Training (Validation Script)
Temporal Bone HRCT Project

Simplified validation script for pipeline testing with limited dataset:
- 2-head architecture (cholesteatoma + ossicular only)
- Reduced epochs (50) and batch size (4)
- Focus on verifying training loop works correctly

Usage:
    python -m pipeline.phase4_model_training_validation \
        --split_dir dataset_splits_validation \
        --roi_dir roi_extracted \
        --labels_csv labels.csv \
        --output_dir models_validation \
        --fold 0 \
        --epochs 50
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.resnet3d import TemporalBoneClassifier
from models.losses import MaskedMultiTaskLoss, compute_class_weights
from data.dataset import TemporalBoneDataset, load_fold_data
from data.transforms import get_train_transforms, get_val_transforms
from utils.download_weights import download_medicalnet_weights

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compute_auc(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> float:
    """
    Compute AUC-ROC for valid samples.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted probabilities
        mask: Valid sample mask
        
    Returns:
        AUC score or 0.5 if insufficient data
    """
    from sklearn.metrics import roc_auc_score
    
    # Filter valid samples
    valid_idx = mask > 0.5
    if valid_idx.sum() < 2:
        return float(0.5)
    
    y_true_valid = y_true[valid_idx]
    y_pred_valid = y_pred[valid_idx]
    
    # Check if we have both classes
    if len(np.unique(y_true_valid)) < 2:
        return float(0.5)
    
    try:
        return float(roc_auc_score(y_true_valid, y_pred_valid))
    except:
        return float(0.5)


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    grad_accumulation_steps: int = 1
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    loss_chole = 0.0
    loss_ossic = 0.0
    n_batches = 0
    
    optimizer.zero_grad()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        labels = batch['labels'].to(device)
        masks = batch['mask'].to(device)
        
        # Forward pass
        outputs = model(images)
        loss, loss_dict = criterion(outputs, labels, masks)
        
        # Backward pass with gradient accumulation
        loss = loss / grad_accumulation_steps
        loss.backward()
        
        if (batch_idx + 1) % grad_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        # Accumulate metrics
        total_loss += loss_dict['loss_total']
        loss_chole += loss_dict['loss_chole']
        loss_ossic += loss_dict['loss_ossic']
        n_batches += 1
        
        pbar.set_postfix({'loss': f"{loss_dict['loss_total']:.4f}"})
    
    # Final optimizer step if needed
    if n_batches % grad_accumulation_steps != 0:
        optimizer.step()
        optimizer.zero_grad()
    
    return {
        'loss': total_loss / max(n_batches, 1),
        'loss_chole': loss_chole / max(n_batches, 1),
        'loss_ossic': loss_ossic / max(n_batches, 1)
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """Validate the model."""
    model.eval()
    
    total_loss = 0.0
    loss_chole = 0.0
    loss_ossic = 0.0
    n_batches = 0
    
    all_preds = []
    all_labels = []
    all_masks = []
    all_ear_ids = []
    
    pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]", leave=False)
    
    for batch in pbar:
        images = batch['image'].to(device)
        labels = batch['labels'].to(device)
        masks = batch['mask'].to(device)
        
        # Forward pass
        outputs = model(images)
        loss, loss_dict = criterion(outputs, labels, masks)
        
        # Accumulate metrics
        total_loss += loss_dict['loss_total']
        loss_chole += loss_dict['loss_chole']
        loss_ossic += loss_dict['loss_ossic']
        n_batches += 1
        
        # Store predictions
        probs = torch.sigmoid(outputs).cpu().numpy()
        all_preds.append(probs)
        all_labels.append(labels.cpu().numpy())
        all_masks.append(masks.cpu().numpy())
        all_ear_ids.extend(batch['ear_id'])
        
        pbar.set_postfix({'loss': f"{loss_dict['loss_total']:.4f}"})
    
    # Concatenate all predictions
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)
    
    # Compute AUC for each task
    auc_chole = compute_auc(all_labels[:, 0], all_preds[:, 0], all_masks[:, 0])
    auc_ossic = compute_auc(all_labels[:, 1], all_preds[:, 1], all_masks[:, 1])
    
    metrics = {
        'loss': total_loss / max(n_batches, 1),
        'loss_chole': loss_chole / max(n_batches, 1),
        'loss_ossic': loss_ossic / max(n_batches, 1),
        'auc_chole': auc_chole,
        'auc_ossic': auc_ossic,
        'auc_mean': (auc_chole + auc_ossic) / 2
    }
    
    predictions = {
        'ear_ids': all_ear_ids,
        'preds': all_preds,
        'labels': all_labels,
        'masks': all_masks
    }
    
    return metrics, predictions


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    path: Path
):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }, path)


def train_fold(
    fold: int,
    config: Dict,
    device: torch.device
) -> Dict:
    """Train a single fold."""
    
    # Setup paths
    fold_path = Path(config['split_dir']) / f"fold_{fold}.json"
    output_dir = Path(config['output_dir']) / f"fold_{fold}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging to file
    log_path = output_dir / 'training.log'
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info(f"=" * 60)
    logger.info(f"Training Fold {fold}")
    logger.info(f"=" * 60)
    logger.info(f"Config: {json.dumps(config, indent=2)}")
    
    # Load data
    train_ear_ids, val_ear_ids, labels_df = load_fold_data(
        str(fold_path),
        config['roi_dir'],
        config['labels_csv']
    )
    
    logger.info(f"Train samples: {len(train_ear_ids)}, Val samples: {len(val_ear_ids)}")
    
    # Create datasets
    train_transforms = get_train_transforms()
    val_transforms = get_val_transforms()
    
    train_dataset = TemporalBoneDataset(
        ear_ids=train_ear_ids,
        roi_dir=config['roi_dir'],
        labels_df=labels_df,
        transforms=train_transforms,
        num_tasks=2
    )
    
    val_dataset = TemporalBoneDataset(
        ear_ids=val_ear_ids,
        roi_dir=config['roi_dir'],
        labels_df=labels_df,
        transforms=val_transforms,
        num_tasks=2
    )
    
    logger.info(f"Train dataset size: {len(train_dataset)}, Val dataset size: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=False  # Disabled to reduce memory pressure
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=False  # Disabled to reduce memory pressure
    )
    
    # Compute class weights
    pos_weights = compute_class_weights(
        labels_df[labels_df['exclusion_status'] == 'include'],
        ['cholesteatoma', 'ossicular_discontinuity']
    )
    logger.info(f"Positive class weights: {pos_weights}")
    
    # Create model
    model = TemporalBoneClassifier(
        num_tasks=2,
        use_cbam=config['use_cbam'],
        pretrained_path=config.get('pretrained_path')
    )
    model = model.to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create loss function
    criterion = MaskedMultiTaskLoss(
        pos_weights=pos_weights,
        num_tasks=2
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config['epochs']
    )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_auc_chole': [],
        'val_auc_ossic': [],
        'val_auc_mean': [],
        'learning_rate': []
    }
    
    best_auc = 0.0
    best_epoch = 0
    patience_counter = 0
    
    # Training loop
    for epoch in range(1, config['epochs'] + 1):
        logger.info(f"\nEpoch {epoch}/{config['epochs']}")
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            grad_accumulation_steps=config['grad_accumulation_steps']
        )
        
        # Validate
        val_metrics, predictions = validate(model, val_loader, criterion, device, epoch)
        
        # Update scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Log metrics
        logger.info(f"  Train Loss: {train_metrics['loss']:.4f}")
        logger.info(f"  Val Loss: {val_metrics['loss']:.4f}")
        logger.info(f"  Val AUC Chole: {val_metrics['auc_chole']:.4f}")
        logger.info(f"  Val AUC Ossic: {val_metrics['auc_ossic']:.4f}")
        logger.info(f"  Val AUC Mean: {val_metrics['auc_mean']:.4f}")
        logger.info(f"  LR: {current_lr:.6f}")
        
        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_auc_chole'].append(val_metrics['auc_chole'])
        history['val_auc_ossic'].append(val_metrics['auc_ossic'])
        history['val_auc_mean'].append(val_metrics['auc_mean'])
        history['learning_rate'].append(current_lr)
        
        # Check for best model
        if val_metrics['auc_mean'] > best_auc:
            best_auc = val_metrics['auc_mean']
            best_epoch = epoch
            patience_counter = 0
            
            # Save best checkpoint
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                output_dir / 'best_checkpoint.pth'
            )
            logger.info(f"  New best model saved! AUC: {best_auc:.4f}")
            
            # Save validation predictions
            pred_df = pd.DataFrame({
                'ear_id': predictions['ear_ids'],
                'prob_chole': predictions['preds'][:, 0],
                'prob_ossic': predictions['preds'][:, 1],
                'label_chole': predictions['labels'][:, 0],
                'label_ossic': predictions['labels'][:, 1],
                'mask_chole': predictions['masks'][:, 0],
                'mask_ossic': predictions['masks'][:, 1]
            })
            pred_df.to_csv(output_dir / 'validation_predictions.csv', index=False)
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= config['early_stopping_patience']:
            logger.info(f"Early stopping triggered at epoch {epoch}")
            break
    
    # Save training history
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save config
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"\nTraining complete!")
    logger.info(f"Best epoch: {best_epoch}, Best AUC: {best_auc:.4f}")
    
    # Remove file handler
    logger.removeHandler(file_handler)
    
    return {
        'fold': fold,
        'best_epoch': best_epoch,
        'best_auc': best_auc,
        'history': history
    }


def main():
    parser = argparse.ArgumentParser(
        description='Phase 4: Model Training (Validation)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Data paths
    parser.add_argument('--split_dir', type=str, default='dataset_splits_validation',
                        help='Directory containing split JSON files')
    parser.add_argument('--roi_dir', type=str, default='roi_extracted',
                        help='Directory containing extracted ROIs')
    parser.add_argument('--labels_csv', type=str, default='labels.csv',
                        help='Path to labels CSV file')
    parser.add_argument('--output_dir', type=str, default='training_checkpoints_validation',
                        help='Output directory for models and logs')
    
    # Training parameters
    parser.add_argument('--fold', type=int, default=None,
                        help='Fold to train (default: all folds)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3,
                        help='Weight decay')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--grad_accumulation_steps', type=int, default=2,
                        help='Gradient accumulation steps')
    
    # Model parameters
    parser.add_argument('--use_cbam', action='store_true', default=True,
                        help='Use CBAM attention')
    parser.add_argument('--pretrained_path', type=str, default='pretrained/resnet_18_23dataset.pth',
                        help='Path to MedicalNet pretrained weights (use None to train from scratch)')
    
    # Hardware
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU device ID (default: auto-detect, use -1 for CPU)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of data loading workers')
    
    args = parser.parse_args()
    
    # Setup device - auto-detect GPU by default
    if args.gpu == -1:
        device = torch.device('cpu')
        logger.info("Using CPU (forced)")
    elif torch.cuda.is_available():
        gpu_id = args.gpu if args.gpu is not None else 0
        device = torch.device(f'cuda:{gpu_id}')
        logger.info(f"Using GPU: {torch.cuda.get_device_name(gpu_id)}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU (no GPU available)")
    
    # Build config (normalize pretrained_path: string "None" or empty => no pretrained)
    pretrained_path = args.pretrained_path
    if isinstance(pretrained_path, str) and pretrained_path.strip().lower() in ('', 'none'):
        pretrained_path = None
    config = {
        'split_dir': args.split_dir,
        'roi_dir': args.roi_dir,
        'labels_csv': args.labels_csv,
        'output_dir': args.output_dir,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'early_stopping_patience': args.early_stopping_patience,
        'grad_accumulation_steps': args.grad_accumulation_steps,
        'use_cbam': args.use_cbam,
        'pretrained_path': pretrained_path,
        'num_workers': args.num_workers,
        'device': str(device),
        'timestamp': datetime.now().isoformat()
    }

    # Ensure MedicalNet pretrained weights exist (auto-download if missing)
    if config['pretrained_path'] is not None:
        p_path = Path(config['pretrained_path'])
        if not p_path.exists():
            output_dir = str(p_path.parent) if p_path.parent != Path('.') else 'pretrained'
            filename = p_path.name if p_path.name else 'resnet_18_23dataset.pth'
            logger.info(f"MedicalNet weights not found at {config['pretrained_path']}; downloading...")
            resolved = download_medicalnet_weights(output_dir=output_dir, filename=filename)
            config['pretrained_path'] = resolved
            logger.info(f"Using MedicalNet weights at {resolved}")

    # Determine which folds to train
    if args.fold is not None:
        folds = [args.fold]
    else:
        # Find all folds
        split_dir = Path(args.split_dir)
        fold_files = sorted(split_dir.glob('fold_*.json'))
        folds = [int(f.stem.split('_')[1]) for f in fold_files]
    
    logger.info(f"Training folds: {folds}")
    
    # Train each fold
    results = []
    for fold in folds:
        result = train_fold(fold, config, device)
        results.append(result)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 60)
    
    aucs = [r['best_auc'] for r in results]
    for r in results:
        logger.info(f"Fold {r['fold']}: Best AUC = {r['best_auc']:.4f} (Epoch {r['best_epoch']})")
    
    if len(aucs) > 1:
        logger.info(f"\nMean AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    
    # Save overall results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'training_summary.json', 'w') as f:
        json.dump({
            'folds': [r['fold'] for r in results],
            'best_aucs': aucs,
            'mean_auc': float(np.mean(aucs)),
            'std_auc': float(np.std(aucs)) if len(aucs) > 1 else 0.0,
            'config': config
        }, f, indent=2)
    
    logger.info("\nPhase 4 (Validation) training completed!")


if __name__ == '__main__':
    main()
