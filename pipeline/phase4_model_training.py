"""
Phase 4: Model Training (Production Script)
Temporal Bone HRCT Project

Full-featured production script for complete dataset (100-120 patients):
- 4-head architecture (cholesteatoma + ossicular + facial_nerve + lscc)
- 150 epochs with early stopping
- Mixed precision training support
- Comprehensive logging and checkpointing

Usage:
    python -m pipeline.phase4_model_training \
        --split_dir dataset_splits \
        --roi_dir roi_extracted \
        --labels_csv labels.csv \
        --output_dir models \
        --fold 0 \
        --epochs 150
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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
    """Compute AUC-ROC for valid samples."""
    from sklearn.metrics import roc_auc_score
    
    valid_idx = mask > 0.5
    if valid_idx.sum() < 2:
        return float(0.5)
    
    y_true_valid = y_true[valid_idx]
    y_pred_valid = y_pred[valid_idx]
    
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
    grad_accumulation_steps: int = 1,
    scaler: Optional[Any] = None,
    use_amp: bool = False
) -> Dict[str, float]:
    """Train for one epoch with optional mixed precision."""
    model.train()
    
    total_loss = 0.0
    loss_chole = 0.0
    loss_ossic = 0.0
    loss_facial = 0.0
    loss_lscc = 0.0
    n_batches = 0
    
    optimizer.zero_grad()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]", leave=False)
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        labels = batch['labels'].to(device)
        masks = batch['mask'].to(device)
        
        # Forward pass with optional AMP
        if use_amp and scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss, loss_dict = criterion(outputs, labels, masks)
            
            # Backward pass
            loss = loss / grad_accumulation_steps
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % grad_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            outputs = model(images)
            loss, loss_dict = criterion(outputs, labels, masks)
            
            loss = loss / grad_accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
        
        # Accumulate metrics
        total_loss += loss_dict['loss_total']
        loss_chole += loss_dict['loss_chole']
        loss_ossic += loss_dict['loss_ossic']
        loss_facial += loss_dict.get('loss_facial', 0.0)
        loss_lscc += loss_dict.get('loss_lscc', 0.0)
        n_batches += 1
        
        pbar.set_postfix({'loss': f"{loss_dict['loss_total']:.4f}"})
    
    # Final optimizer step if needed
    if n_batches % grad_accumulation_steps != 0:
        if use_amp and scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()
    
    return {
        'loss': total_loss / max(n_batches, 1),
        'loss_chole': loss_chole / max(n_batches, 1),
        'loss_ossic': loss_ossic / max(n_batches, 1),
        'loss_facial': loss_facial / max(n_batches, 1),
        'loss_lscc': loss_lscc / max(n_batches, 1)
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    num_tasks: int = 4
) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    """Validate the model."""
    model.eval()
    
    total_loss = 0.0
    loss_chole = 0.0
    loss_ossic = 0.0
    loss_facial = 0.0
    loss_lscc = 0.0
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
        
        outputs = model(images)
        loss, loss_dict = criterion(outputs, labels, masks)
        
        total_loss += loss_dict['loss_total']
        loss_chole += loss_dict['loss_chole']
        loss_ossic += loss_dict['loss_ossic']
        loss_facial += loss_dict.get('loss_facial', 0.0)
        loss_lscc += loss_dict.get('loss_lscc', 0.0)
        n_batches += 1
        
        probs = torch.sigmoid(outputs).cpu().numpy()
        all_preds.append(probs)
        all_labels.append(labels.cpu().numpy())
        all_masks.append(masks.cpu().numpy())
        all_ear_ids.extend(batch['ear_id'])
        
        pbar.set_postfix({'loss': f"{loss_dict['loss_total']:.4f}"})
    
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
        'auc_ossic': auc_ossic
    }
    
    auc_components = [auc_chole, auc_ossic]

    if num_tasks >= 3:
        auc_facial = compute_auc(all_labels[:, 2], all_preds[:, 2], all_masks[:, 2])
        metrics['loss_facial'] = loss_facial / max(n_batches, 1)
        metrics['auc_facial'] = auc_facial
        auc_components.append(auc_facial)

    if num_tasks >= 4:
        auc_lscc = compute_auc(all_labels[:, 3], all_preds[:, 3], all_masks[:, 3])
        metrics['loss_lscc'] = loss_lscc / max(n_batches, 1)
        metrics['auc_lscc'] = auc_lscc
        auc_components.append(auc_lscc)

    metrics['auc_mean'] = float(np.mean(auc_components))
    
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
    path: Path,
    scheduler = None
):
    """Save model checkpoint."""
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    if scheduler is not None:
        state['scheduler_state_dict'] = scheduler.state_dict()
    torch.save(state, path)


def train_fold(
    fold: int,
    config: Dict,
    device: torch.device
) -> Dict:
    """Train a single fold."""
    
    num_tasks = config.get('num_tasks', 4)
    
    # Setup paths
    fold_path = Path(config['split_dir']) / f"fold_{fold}.json"
    output_dir = Path(config['output_dir']) / f"fold_{fold}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_path = output_dir / 'training.log'
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    logger.info(f"=" * 60)
    logger.info(f"Training Fold {fold} (Production)")
    logger.info(f"=" * 60)
    
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
        num_tasks=num_tasks
    )
    
    val_dataset = TemporalBoneDataset(
        ear_ids=val_ear_ids,
        roi_dir=config['roi_dir'],
        labels_df=labels_df,
        transforms=val_transforms,
        num_tasks=num_tasks
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # Compute class weights
    label_cols = ['cholesteatoma', 'ossicular_discontinuity']
    if num_tasks >= 3:
        label_cols.append('facial_dehiscence')
    if num_tasks >= 4:
        label_cols.append('lscc_dehiscence')
    
    pos_weights = compute_class_weights(
        labels_df[labels_df['exclusion_status'] == 'include'],
        label_cols
    )
    logger.info(f"Positive class weights: {pos_weights}")
    
    # Create model
    model = TemporalBoneClassifier(
        num_tasks=num_tasks,
        use_cbam=config['use_cbam'],
        pretrained_path=config.get('pretrained_path')
    )
    model = model.to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create loss function
    criterion = MaskedMultiTaskLoss(
        pos_weights=pos_weights,
        num_tasks=num_tasks
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
    
    # Mixed precision scaler
    use_amp = config.get('mixed_precision', False) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_auc_chole': [],
        'val_auc_ossic': [],
        'val_auc_mean': [],
        'learning_rate': []
    }
    if num_tasks >= 3:
        history['val_auc_facial'] = []
    if num_tasks >= 4:
        history['val_auc_lscc'] = []
    
    best_auc = 0.0
    best_epoch = 0
    patience_counter = 0
    
    # Training loop
    for epoch in range(1, config['epochs'] + 1):
        logger.info(f"\nEpoch {epoch}/{config['epochs']}")
        
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            grad_accumulation_steps=config['grad_accumulation_steps'],
            scaler=scaler,
            use_amp=use_amp
        )
        
        val_metrics, predictions = validate(
            model, val_loader, criterion, device, epoch, num_tasks
        )
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        logger.info(f"  Train Loss: {train_metrics['loss']:.4f}")
        logger.info(f"  Val Loss: {val_metrics['loss']:.4f}")
        logger.info(f"  Val AUC Mean: {val_metrics['auc_mean']:.4f}")
        
        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_auc_chole'].append(val_metrics['auc_chole'])
        history['val_auc_ossic'].append(val_metrics['auc_ossic'])
        history['val_auc_mean'].append(val_metrics['auc_mean'])
        history['learning_rate'].append(current_lr)
        if num_tasks >= 3:
            history['val_auc_facial'].append(val_metrics.get('auc_facial', 0.5))
        if num_tasks >= 4:
            history['val_auc_lscc'].append(val_metrics.get('auc_lscc', 0.5))
        
        if val_metrics['auc_mean'] > best_auc:
            best_auc = val_metrics['auc_mean']
            best_epoch = epoch
            patience_counter = 0
            
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                output_dir / 'best_checkpoint.pth',
                scheduler
            )
            logger.info(f"  New best model! AUC: {best_auc:.4f}")
            
            # Save predictions
            pred_cols = ['ear_id', 'prob_chole', 'prob_ossic']
            pred_data = {
                'ear_id': predictions['ear_ids'],
                'prob_chole': predictions['preds'][:, 0],
                'prob_ossic': predictions['preds'][:, 1]
            }
            if num_tasks >= 3:
                pred_cols.append('prob_facial')
                pred_data['prob_facial'] = predictions['preds'][:, 2]
            if num_tasks >= 4:
                pred_cols.append('prob_lscc')
                pred_data['prob_lscc'] = predictions['preds'][:, 3]
            
            pd.DataFrame(pred_data).to_csv(
                output_dir / 'validation_predictions.csv', index=False
            )
        else:
            patience_counter += 1
        
        if patience_counter >= config['early_stopping_patience']:
            logger.info(f"Early stopping at epoch {epoch}")
            break
    
    # Save history and config
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"\nBest epoch: {best_epoch}, Best AUC: {best_auc:.4f}")
    logger.removeHandler(file_handler)
    
    return {
        'fold': fold,
        'best_epoch': best_epoch,
        'best_auc': best_auc,
        'history': history
    }


def main():
    parser = argparse.ArgumentParser(
        description='Phase 4: Model Training (Production)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--split_dir', type=str, default='dataset_splits')
    parser.add_argument('--roi_dir', type=str, default='roi_extracted')
    parser.add_argument('--labels_csv', type=str, default='labels.csv')
    parser.add_argument('--output_dir', type=str, default='training_checkpoints')
    
    parser.add_argument('--fold', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--early_stopping_patience', type=int, default=15)
    parser.add_argument('--grad_accumulation_steps', type=int, default=2)
    parser.add_argument('--num_tasks', type=int, default=4)
    
    parser.add_argument('--use_cbam', action='store_true', default=True)
    parser.add_argument('--pretrained_path', type=str, default='pretrained/resnet_18_23dataset.pth',
                        help='Path to MedicalNet pretrained weights (use None to train from scratch)')
    parser.add_argument('--mixed_precision', action='store_true', default=True)
    
    parser.add_argument('--gpu', type=int, default=None,
                        help='GPU device ID (default: auto-detect, use -1 for CPU)')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of DataLoader workers (use 0 to avoid memory issues on limited RAM systems)')
    
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
    
    config = vars(args)
    # Normalize pretrained_path: string "None" or empty => no pretrained
    p = config.get('pretrained_path')
    if isinstance(p, str) and p.strip().lower() in ('', 'none'):
        config['pretrained_path'] = None
    config['device'] = str(device)
    config['timestamp'] = datetime.now().isoformat()

    # Ensure MedicalNet pretrained weights exist (auto-download if missing)
    pretrained_path = config.get('pretrained_path')
    if pretrained_path is not None:
        p_path = Path(pretrained_path)
        if not p_path.exists():
            output_dir = str(p_path.parent) if p_path.parent != Path('.') else 'pretrained'
            filename = p_path.name if p_path.name else 'resnet_18_23dataset.pth'
            logger.info(f"MedicalNet weights not found at {pretrained_path}; downloading...")
            resolved = download_medicalnet_weights(output_dir=output_dir, filename=filename)
            config['pretrained_path'] = resolved
            logger.info(f"Using MedicalNet weights at {resolved}")

    if args.fold is not None:
        folds = [args.fold]
    else:
        split_dir = Path(args.split_dir)
        fold_files = sorted(split_dir.glob('fold_*.json'))
        folds = [int(f.stem.split('_')[1]) for f in fold_files]
    
    logger.info(f"Training folds: {folds}")
    
    results = []
    for fold in folds:
        result = train_fold(fold, config, device)
        results.append(result)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING SUMMARY (PRODUCTION)")
    logger.info("=" * 60)
    
    aucs = [r['best_auc'] for r in results]
    for r in results:
        logger.info(f"Fold {r['fold']}: Best AUC = {r['best_auc']:.4f} (Epoch {r['best_epoch']})")
    
    if len(aucs) > 1:
        logger.info(f"\nMean AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    
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
    
    logger.info("\nPhase 4 (Production) training completed!")


if __name__ == '__main__':
    main()
