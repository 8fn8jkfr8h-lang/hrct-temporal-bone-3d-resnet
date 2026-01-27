"""
Batch Viewer for Processed DICOM Data
Generates overview images for all patients

Usage:
    python -m utils.viewer_batch
    
    Or from Python:
    from utils.viewer_batch import create_patient_overview
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import sys


def create_patient_overview(patient_dir: Path, output_dir: Path):
    """Create overview image for one patient"""
    
    patient_id = patient_dir.name
    
    # Create 4 rows x 3 columns grid
    # Rows 0-1: Left ear (3 axial + 3 coronal)
    # Rows 2-3: Right ear (3 axial + 3 coronal)
    fig, axes = plt.subplots(4, 3, figsize=(15, 20))
    fig.suptitle(f'{patient_id} - Overview', fontsize=16, fontweight='bold')
    
    for side_idx, side in enumerate(['left', 'right']):
        base_row = side_idx * 2  # 0 for left, 2 for right
        side_dir = patient_dir / side
        
        if not side_dir.exists():
            # Hide all axes for this side if data doesn't exist
            for row in range(2):
                for col in range(3):
                    axes[base_row + row, col].axis('off')
            continue
        
        # Load data
        axial = np.load(side_dir / 'axial_volume.npy')
        coronal = np.load(side_dir / 'coronal_volume.npy')
        
        # Flip coronal view vertically (axis 1) so Head is Up
        # Data is (Y, Z, X), we flip Z
        coronal = np.flip(coronal, axis=1)
        
        with open(side_dir / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        # Get middle slice indices
        axial_mid = axial.shape[0] // 2
        coronal_mid = coronal.shape[0] // 2
        
        # Row 0 (or 2): 3 Axial slices
        axial_offsets = [-10, 0, 10]
        for col, offset in enumerate(axial_offsets):
            slice_idx = np.clip(axial_mid + offset, 0, axial.shape[0] - 1)
            axes[base_row, col].imshow(axial[slice_idx], cmap='bone', vmin=0, vmax=1)
            axes[base_row, col].set_title(f'{side.upper()} - Axial {slice_idx}', fontsize=10)
            axes[base_row, col].axis('off')
        
        # Row 1 (or 3): 3 Coronal slices
        coronal_offsets = [-20, 0, 20]
        for col, offset in enumerate(coronal_offsets):
            slice_idx = np.clip(coronal_mid + offset, 0, coronal.shape[0] - 1)
            axes[base_row + 1, col].imshow(coronal[slice_idx], cmap='bone', vmin=0, vmax=1)
            axes[base_row + 1, col].set_title(f'{side.upper()} - Coronal {slice_idx}', fontsize=10)
            axes[base_row + 1, col].axis('off')
    
    plt.tight_layout()
    
    # Save
    output_file = output_dir / f'{patient_id}_overview.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {output_file.name}")


def main():
    """Main function"""
    
    data_dir = Path('processed_data')
    output_dir = Path('visualizations')
    output_dir.mkdir(exist_ok=True)
    
    print("="*60)
    print("BATCH VIEWER - Generating Overview Images")
    print("="*60)
    
    patients = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    
    print(f"\nProcessing {len(patients)} patients...")
    
    for patient_dir in patients:
        print(f"\n{patient_dir.name}:")
        create_patient_overview(patient_dir, output_dir)
    
    print(f"\n{'='*60}")
    print(f"Complete! Images saved to: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
