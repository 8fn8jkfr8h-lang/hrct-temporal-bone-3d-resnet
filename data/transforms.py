"""
Data augmentation transforms for temporal bone HRCT.

Uses MONAI transforms for 3D medical image augmentation.
"""

import numpy as np
import warnings

# Suppress setuptools deprecation warning from MONAI
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='pkg_resources is deprecated')

try:
    from monai.transforms.compose import Compose
    from monai.transforms.utility.dictionary import EnsureChannelFirstd, ToTensord
    from monai.transforms.spatial.dictionary import RandAffined
    from monai.transforms.intensity.dictionary import (
        RandGaussianNoised,
        RandGaussianSmoothd,
        RandScaleIntensityd,
        RandCoarseDropoutd,
        ScaleIntensityd
    )
    MONAI_AVAILABLE = True
except ImportError:
    MONAI_AVAILABLE = False
    print("Warning: MONAI not installed. Using basic transforms only.")


def get_train_transforms():
    """
    Get training augmentation pipeline.
    
    Includes:
    - Random affine (rotation, translation, scaling)
    - Intensity augmentation (scaling, noise, smoothing)
    - Coarse dropout (cutout regularization)
    """
    if not MONAI_AVAILABLE:
        return BasicTrainTransform()
    
    return Compose([
        # Ensure channel first format
        EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
        
        # Normalize intensity to [0, 1]
        ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
        
        # Geometric augmentation
        RandAffined(
            keys=["image"],
            mode="bilinear",
            prob=0.8,
            rotate_range=[0.26, 0.26, 0.26],  # ±15° in radians
            translate_range=[10, 10, 10],      # voxels
            scale_range=[0.1, 0.1, 0.1],       # ±10%
            padding_mode="zeros"
        ),
        
        # Intensity augmentation
        RandScaleIntensityd(
            keys=["image"],
            factors=0.2,      # ±20% intensity scaling
            prob=0.5
        ),
        
        RandGaussianNoised(
            keys=["image"],
            std=0.02,
            prob=0.5
        ),
        
        RandGaussianSmoothd(
            keys=["image"],
            sigma_x=(0.5, 1.0),
            sigma_y=(0.5, 1.0),
            sigma_z=(0.5, 1.0),
            prob=0.3
        ),
        
        # Coarse dropout (3D cutout)
        RandCoarseDropoutd(
            keys=["image"],
            holes=5,
            spatial_size=[16, 16, 16],
            fill_value=0,
            prob=0.2
        ),
        
        # Convert to tensor
        ToTensord(keys=["image"])
    ])


def get_val_transforms():
    """
    Get validation/test transforms (no augmentation).
    
    Only applies normalization and tensor conversion.
    """
    if not MONAI_AVAILABLE:
        return BasicValTransform()
    
    return Compose([
        # Ensure channel first format
        EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
        
        # Normalize intensity to [0, 1]
        ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
        
        # Convert to tensor
        ToTensord(keys=["image"])
    ])


class BasicTrainTransform:
    """Fallback training transform when MONAI is not available."""
    
    def __call__(self, data: dict) -> dict:
        import torch
        
        image = data["image"]
        
        # Add channel dimension if needed
        if image.ndim == 3:
            image = image[np.newaxis, ...]
        
        # Basic normalization
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        # Random flip (simple augmentation)
        if np.random.random() > 0.5:
            image = np.flip(image, axis=1).copy()
        
        data["image"] = torch.from_numpy(image.astype(np.float32))
        return data


class BasicValTransform:
    """Fallback validation transform when MONAI is not available."""
    
    def __call__(self, data: dict) -> dict:
        import torch
        
        image = data["image"]
        
        # Add channel dimension if needed
        if image.ndim == 3:
            image = image[np.newaxis, ...]
        
        # Basic normalization
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        data["image"] = torch.from_numpy(image.astype(np.float32))
        return data


if __name__ == "__main__":
    
    print(f"MONAI available: {MONAI_AVAILABLE}")
    
    # Test transforms
    train_tf = get_train_transforms()
    val_tf = get_val_transforms()
    
    # Create dummy data
    sample = {"image": np.random.randn(128, 128, 128).astype(np.float32)}
    
    train_out = train_tf(sample.copy())
    print(f"Train transform output shape: {train_out['image'].shape}")
    
    val_out = val_tf(sample.copy())
    print(f"Val transform output shape: {val_out['image'].shape}")
