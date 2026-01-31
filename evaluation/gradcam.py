"""
Grad-CAM implementation for 3D CNNs.

Provides interpretability tools for visualizing model attention
on volumetric medical images.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from pathlib import Path
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


class GradCAM3D:
    """
    Grad-CAM for 3D convolutional neural networks.
    
    Generates class activation maps showing which regions of the input
    volume most influenced the model's prediction.
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
        target_layer_name: Optional[str] = None
    ):
        """
        Initialize Grad-CAM.
        
        Args:
            model: The neural network model
            target_layer: The layer to compute gradients for (usually last conv layer)
            target_layer_name: Name of target layer (alternative to target_layer)
        """
        self.model = model
        self.model.eval()
        
        # Find target layer
        if target_layer is not None:
            self.target_layer = target_layer
        elif target_layer_name is not None:
            self.target_layer = self._find_layer(target_layer_name)
        else:
            # Default: try to find the last convolutional layer
            self.target_layer = self._find_last_conv_layer()
        
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)
    
    def _find_layer(self, name: str) -> nn.Module:
        """Find a layer by name."""
        for n, module in self.model.named_modules():
            if n == name:
                return module
        raise ValueError(f"Layer '{name}' not found in model")
    
    def _find_last_conv_layer(self) -> nn.Module:
        """Find the last convolutional layer in the model."""
        last_conv = None
        for module in self.model.modules():
            if isinstance(module, (nn.Conv3d, nn.Conv2d)):
                last_conv = module
        
        if last_conv is None:
            raise ValueError("No convolutional layer found in model")
        
        return last_conv
    
    def _save_activation(self, module: nn.Module, input: torch.Tensor, output: torch.Tensor):
        """Hook to save forward activations."""
        self.activations = output.detach()
    
    def _save_gradient(self, module: nn.Module, grad_input, grad_output):
        """Hook to save backward gradients."""
        self.gradients = grad_output[0].detach()
    
    def __call__(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: Input volume tensor (B, C, D, H, W)
            target_class: Class index to compute gradients for (default: predicted class)
            
        Returns:
            Grad-CAM heatmap (D, H, W) normalized to [0, 1]
        """
        # Ensure model is in eval mode
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Determine target class
        if target_class is None:
            target_class = output.argmax(dim=1).item() if output.dim() > 1 else 0
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        if output.dim() > 1 and output.shape[1] > 1:
            one_hot = torch.zeros_like(output)
            one_hot[0, target_class] = 1
            output.backward(gradient=one_hot, retain_graph=True)
        else:
            # Single output (binary classification)
            output.backward(retain_graph=True)
        
        # Get weights: global average pool of gradients
        weights = self.gradients.mean(dim=(2, 3, 4), keepdim=True)  # (B, C, 1, 1, 1)
        
        # Weighted sum of activations
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (B, 1, D, H, W)
        
        # ReLU to keep only positive contributions
        cam = torch.relu(cam)
        
        # Remove batch and channel dimensions
        cam = cam.squeeze().cpu().numpy()
        
        # Upsample to input size
        from scipy.ndimage import zoom
        input_shape = input_tensor.shape[2:]  # (D, H, W)
        zoom_factors = tuple(s / c for s, c in zip(input_shape, cam.shape))
        cam = zoom(cam, zoom_factors, order=1)
        
        # Normalize to [0, 1]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam


def generate_gradcam_overlay(
    volume: np.ndarray,
    heatmap: np.ndarray,
    slice_idx: int,
    axis: int = 0,
    alpha: float = 0.4,
    colormap: str = 'jet'
) -> np.ndarray:
    """
    Create overlay of Grad-CAM heatmap on CT slice.
    
    Args:
        volume: Original CT volume (D, H, W)
        heatmap: Grad-CAM heatmap (D, H, W)
        slice_idx: Index of slice to visualize
        axis: Axis to slice (0=axial, 1=sagittal, 2=coronal)
        alpha: Transparency of heatmap overlay
        colormap: Matplotlib colormap for heatmap
        
    Returns:
        RGB overlay image (H, W, 3)
    """
    # Get slices
    if axis == 0:
        ct_slice = volume[slice_idx, :, :]
        heat_slice = heatmap[slice_idx, :, :]
    elif axis == 1:
        ct_slice = volume[:, slice_idx, :]
        heat_slice = heatmap[:, slice_idx, :]
    else:
        ct_slice = volume[:, :, slice_idx]
        heat_slice = heatmap[:, :, slice_idx]
    
    # Normalize CT slice
    ct_slice = (ct_slice - ct_slice.min()) / (ct_slice.max() - ct_slice.min() + 1e-8)
    
    # Convert CT to RGB (grayscale)
    ct_rgb = np.stack([ct_slice] * 3, axis=-1)
    
    # Apply colormap to heatmap
    cmap = plt.get_cmap(colormap)
    heat_rgb = cmap(heat_slice)[:, :, :3]  # Remove alpha channel
    
    # Blend
    overlay = (1 - alpha) * ct_rgb + alpha * heat_rgb
    overlay = np.clip(overlay, 0, 1)
    
    return (overlay * 255).astype(np.uint8)


def save_gradcam_slices(
    volume: np.ndarray,
    heatmap: np.ndarray,
    output_dir: Path,
    case_id: str,
    prediction: float,
    ground_truth: int,
    n_slices: int = 5
) -> None:
    """
    Save Grad-CAM visualization for a case.
    
    Args:
        volume: Original CT volume (D, H, W)
        heatmap: Grad-CAM heatmap (D, H, W)
        output_dir: Directory to save images
        case_id: Identifier for the case
        prediction: Model's predicted probability
        ground_truth: Ground truth label
        n_slices: Number of slices to save per axis
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine slice indices (evenly spaced through volume)
    d, h, w = volume.shape
    axial_indices = np.linspace(d * 0.2, d * 0.8, n_slices).astype(int)
    
    # Create figure with multiple slices
    fig, axes = plt.subplots(3, n_slices, figsize=(4 * n_slices, 12))
    
    for i, idx in enumerate(axial_indices):
        # Row 0: Original CT
        axes[0, i].imshow(volume[idx, :, :], cmap='gray')
        axes[0, i].set_title(f'Slice {idx}')
        axes[0, i].axis('off')
        
        # Row 1: Grad-CAM heatmap
        axes[1, i].imshow(heatmap[idx, :, :], cmap='jet', vmin=0, vmax=1)
        axes[1, i].set_title('Grad-CAM')
        axes[1, i].axis('off')
        
        # Row 2: Overlay
        overlay = generate_gradcam_overlay(volume, heatmap, idx, axis=0, alpha=0.5)
        axes[2, i].imshow(overlay)
        axes[2, i].set_title('Overlay')
        axes[2, i].axis('off')
    
    # Add row labels
    axes[0, 0].set_ylabel('Original CT', fontsize=12)
    axes[1, 0].set_ylabel('Grad-CAM', fontsize=12)
    axes[2, 0].set_ylabel('Overlay', fontsize=12)
    
    # Add title with prediction info
    pred_label = 'Positive' if prediction > 0.5 else 'Negative'
    gt_label = 'Positive' if ground_truth == 1 else 'Negative'
    result = 'CORRECT' if (prediction > 0.5) == (ground_truth == 1) else 'INCORRECT'
    
    fig.suptitle(f'{case_id}\nPrediction: {pred_label} ({prediction:.3f}) | Ground Truth: {gt_label} | {result}',
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / f'{case_id}_gradcam.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Saved Grad-CAM visualization for {case_id}")


def generate_gradcam_for_batch(
    model: nn.Module,
    dataloader,
    device: torch.device,
    output_dir: Path,
    target_layer_name: Optional[str] = None,
    max_samples: int = 10,
    task_idx: int = 0
) -> None:
    """
    Generate Grad-CAM visualizations for a batch of samples.
    
    Args:
        model: Trained model
        dataloader: DataLoader with samples
        device: Device to run inference on
        output_dir: Directory to save visualizations
        target_layer_name: Name of layer for Grad-CAM (default: auto-detect)
        max_samples: Maximum number of samples to visualize
        task_idx: Index of task/output head to visualize
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        gradcam = GradCAM3D(model, target_layer_name=target_layer_name)
    except ValueError as e:
        logger.error(f"Failed to initialize Grad-CAM: {e}")
        return
    
    model.eval()
    samples_processed = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if samples_processed >= max_samples:
                break
            
            volumes = batch['volume'].to(device)
            labels = batch['labels']
            ear_ids = batch.get('ear_id', [f'sample_{i}' for i in range(len(volumes))])
            
            for i in range(len(volumes)):
                if samples_processed >= max_samples:
                    break
                
                volume = volumes[i:i+1]
                volume.requires_grad_(True)
                
                # Get prediction
                with torch.enable_grad():
                    output = model(volume)
                    if output.dim() > 1 and output.shape[1] > task_idx:
                        pred = torch.sigmoid(output[0, task_idx]).item()
                    else:
                        pred = torch.sigmoid(output[0]).item()
                    
                    # Generate Grad-CAM
                    heatmap = gradcam(volume, target_class=task_idx)
                
                # Get ground truth
                if labels.dim() > 1:
                    gt = int(labels[i, task_idx].item())
                else:
                    gt = int(labels[i].item())
                
                # Get volume as numpy
                vol_np = volume[0, 0].cpu().numpy()
                
                # Save visualization
                ear_id = ear_ids[i] if isinstance(ear_ids, list) else ear_ids
                save_gradcam_slices(vol_np, heatmap, output_dir, ear_id, pred, gt)
                
                samples_processed += 1
    
    logger.info(f"Generated Grad-CAM for {samples_processed} samples")
