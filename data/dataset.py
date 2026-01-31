"""
PyTorch Dataset for Temporal Bone HRCT ROIs.

Loads extracted ROI volumes and corresponding labels for training.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class TemporalBoneDataset(Dataset):
    """
    Dataset for loading temporal bone ROI volumes and labels.
    
    Supports:
    - Loading from fold JSON files
    - Multi-task labels with masking for NULL values
    - Configurable transforms
    """
    
    def __init__(
        self,
        ear_ids: List[str],
        roi_dir: str,
        labels_df: pd.DataFrame,
        transforms: Optional[Callable] = None,
        num_tasks: int = 2
    ):
        """
        Args:
            ear_ids: List of ear IDs (e.g., ['pt_01_R', 'pt_02_L'])
            roi_dir: Directory containing extracted ROIs
            labels_df: DataFrame with labels indexed by patient_id
            transforms: Optional transform pipeline
            num_tasks: Number of classification tasks (2 or 3)
        """
        self.roi_dir = Path(roi_dir)
        self.labels_df = labels_df
        self.transforms = transforms
        self.num_tasks = num_tasks
        
        # Parse ear IDs and build sample list
        self.samples = []
        for ear_id in ear_ids:
            # Parse ear_id format: pt_XX_L or pt_XX_R
            parts = ear_id.rsplit('_', 1)
            if len(parts) != 2:
                continue
            
            patient_id = parts[0]
            ear_code = parts[1]  # 'L' or 'R'
            ear_dir = 'left' if ear_code == 'L' else 'right'
            
            # Check if ROI exists
            roi_path = self.roi_dir / patient_id / ear_dir / 'axial_roi.npy'
            if not roi_path.exists():
                continue
            
            # Get labels
            labels, mask = self._get_labels(patient_id, ear_code)
            
            self.samples.append({
                'ear_id': ear_id,
                'patient_id': patient_id,
                'ear': ear_code,
                'roi_path': roi_path,
                'labels': labels,
                'mask': mask
            })
    
    def _get_labels(self, patient_id: str, ear: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get labels and validity mask for a sample.
        
        Returns:
            Tuple of (labels, mask) arrays
        """
        # Find matching row in labels_df
        mask_df = (self.labels_df['patient_id'] == patient_id) & (self.labels_df['ear'] == ear)
        rows = self.labels_df[mask_df]
        
        if len(rows) == 0:
            # No label found - return zeros with zero mask
            labels = np.zeros(self.num_tasks, dtype=np.float32)
            mask = np.zeros(self.num_tasks, dtype=np.float32)
            return labels, mask
        
        row = rows.iloc[0]
        
        # Extract labels
        labels = []
        mask = []
        
        # Cholesteatoma
        chole = row.get('cholesteatoma', np.nan)
        labels.append(0.0 if pd.isna(chole) else float(chole))
        mask.append(0.0 if pd.isna(chole) else 1.0)
        
        # Ossicular discontinuity
        ossic = row.get('ossicular_discontinuity', np.nan)
        labels.append(0.0 if pd.isna(ossic) else float(ossic))
        mask.append(0.0 if pd.isna(ossic) else 1.0)
        
        # Facial nerve (if 3 tasks)
        if self.num_tasks >= 3:
            facial = row.get('facial_dehiscence', np.nan)
            labels.append(0.0 if pd.isna(facial) else float(facial))
            mask.append(0.0 if pd.isna(facial) else 1.0)
        
        return np.array(labels, dtype=np.float32), np.array(mask, dtype=np.float32)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample.
        
        Returns:
            Dictionary with keys: 'image', 'labels', 'mask', 'ear_id'
        """
        sample = self.samples[idx]
        
        # Load ROI volume
        volume = np.load(sample['roi_path'])
        
        # Prepare data dict for transforms
        data = {
            'image': volume.astype(np.float32),
            'labels': sample['labels'],
            'mask': sample['mask'],
            'ear_id': sample['ear_id']
        }
        
        # Apply transforms
        if self.transforms is not None:
            data = self.transforms(data)
        else:
            # Basic processing if no transforms
            image = data['image']
            if image.ndim == 3:
                image = image[np.newaxis, ...]
            image = (image - image.min()) / (image.max() - image.min() + 1e-8)
            data['image'] = torch.from_numpy(image.astype(np.float32))
        
        # Ensure labels and mask are tensors
        if not isinstance(data['labels'], torch.Tensor):
            data['labels'] = torch.from_numpy(data['labels'])
        if not isinstance(data['mask'], torch.Tensor):
            data['mask'] = torch.from_numpy(data['mask'])
        
        return data


def load_fold_data(
    fold_path: str,
    roi_dir: str,
    labels_csv: str,
    num_tasks: int = 2
) -> Tuple[List[str], List[str], pd.DataFrame]:
    """
    Load train/val ear IDs from a fold JSON file.
    
    Args:
        fold_path: Path to fold JSON file
        roi_dir: Directory containing ROIs
        labels_csv: Path to labels CSV
        num_tasks: Number of classification tasks
        
    Returns:
        Tuple of (train_ear_ids, val_ear_ids, labels_df)
    """
    with open(fold_path, 'r') as f:
        fold_data = json.load(f)
    
    train_ear_ids = fold_data['train_ear_ids']
    val_ear_ids = fold_data['val_ear_ids']
    
    labels_df = pd.read_csv(labels_csv)
    
    return train_ear_ids, val_ear_ids, labels_df


def create_dataloaders(
    fold_path: str,
    roi_dir: str,
    labels_csv: str,
    batch_size: int = 4,
    num_workers: int = 0,
    num_tasks: int = 2,
    train_transforms = None,
    val_transforms = None
):
    """
    Create training and validation DataLoaders from a fold file.
    
    Args:
        fold_path: Path to fold JSON
        roi_dir: Directory containing ROIs
        labels_csv: Path to labels CSV
        batch_size: Batch size
        num_workers: Number of data loading workers
        num_tasks: Number of classification tasks
        train_transforms: Training augmentation pipeline
        val_transforms: Validation transform pipeline
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    from torch.utils.data import DataLoader
    
    train_ear_ids, val_ear_ids, labels_df = load_fold_data(
        fold_path, roi_dir, labels_csv, num_tasks
    )
    
    train_dataset = TemporalBoneDataset(
        ear_ids=train_ear_ids,
        roi_dir=roi_dir,
        labels_df=labels_df,
        transforms=train_transforms,
        num_tasks=num_tasks
    )
    
    val_dataset = TemporalBoneDataset(
        ear_ids=val_ear_ids,
        roi_dir=roi_dir,
        labels_df=labels_df,
        transforms=val_transforms,
        num_tasks=num_tasks
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset loading
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    
    from data.transforms import get_train_transforms, get_val_transforms
    
    # Test with actual data
    fold_path = "dataset_splits_validation/fold_0.json"
    roi_dir = "roi_extracted"
    labels_csv = "labels.csv"
    
    if Path(fold_path).exists():
        train_loader, val_loader = create_dataloaders(
            fold_path=fold_path,
            roi_dir=roi_dir,
            labels_csv=labels_csv,
            batch_size=2,
            num_tasks=2,
            train_transforms=get_train_transforms(),
            val_transforms=get_val_transforms()
        )
        
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
        # Test one batch
        for batch in train_loader:
            print(f"Image shape: {batch['image'].shape}")
            print(f"Labels shape: {batch['labels'].shape}")
            print(f"Mask shape: {batch['mask'].shape}")
            print(f"Ear IDs: {batch['ear_id']}")
            break
    else:
        print(f"Fold file not found: {fold_path}")
        print("Run phase3 stratification first.")
