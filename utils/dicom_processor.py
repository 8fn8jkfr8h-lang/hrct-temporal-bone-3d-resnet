"""
DICOM Processing Utilities
Temporal Bone HRCT Project
"""

import json
import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk
from pathlib import Path
from typing import Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class DICOMProcessor:
    """Process DICOM temporal bone CT scans"""
    
    # Bone windowing parameters from spec
    BONE_WINDOW_WIDTH = 4000
    BONE_WINDOW_LEVEL = 700
    
    # Lateral split parameters
    MIDLINE_MARGIN = 20  # pixels overlap at midline
    
    def __init__(self, input_dir: str, output_dir: str, labels_file: str):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.labels_file = labels_file
        
        # Load labels
        self.labels_df = pd.read_csv(labels_file)
        print(f"Loaded labels for {len(self.labels_df)} ear records")
        
    def load_and_sort_dicom_series(self, patient_dir: Path) -> Tuple[np.ndarray, Dict]:
        """
        Load all DICOM files for a patient and sort by Z-position
        
        Returns:
            volume: 3D numpy array (slices, height, width)
            metadata: dict with spacing, patient info, etc.
        """
        print(f"\n  Loading DICOM files from {patient_dir.name}...")
        
        # Get all DICOM files
        dicom_files = sorted(patient_dir.glob('0_*'))
        
        if len(dicom_files) == 0:
            raise ValueError(f"No DICOM files found in {patient_dir}")
        
        print(f"  Found {len(dicom_files)} DICOM files")
        
        # Load all slices with their Z-position
        slices_with_position = []
        
        for dcm_file in dicom_files:
            try:
                ds = pydicom.dcmread(str(dcm_file))
                z_position = float(ds.ImagePositionPatient[2])
                slices_with_position.append((z_position, ds))
            except Exception as e:
                print(f"  Warning: Could not read {dcm_file.name}: {e}")
                continue
        
        if len(slices_with_position) == 0:
            raise ValueError(f"No valid DICOM files loaded from {patient_dir}")
        
        # Sort by Z-position (ImagePositionPatient[2])
        slices_with_position.sort(key=lambda x: x[0])
        
        print(f"  Sorted {len(slices_with_position)} slices by Z-position")
        print(f"  Z-range: {slices_with_position[0][0]:.2f} to {slices_with_position[-1][0]:.2f} mm")
        
        # Extract metadata from first slice
        first_slice = slices_with_position[0][1]
        
        # Calculate actual Z-spacing from positions
        z_positions = [float(z) for z, _ in slices_with_position]
        if len(z_positions) > 1:
            # Calculate mean spacing between slices
            z_diffs = np.diff(z_positions)
            z_spacing = float(np.mean(np.abs(z_diffs)))
        else:
            z_spacing = float(first_slice.SliceThickness)
            
        print(f"  Calculated Z-spacing: {z_spacing:.4f} mm (Slice Thickness: {first_slice.SliceThickness} mm)")
        
        metadata = {
            'patient_id': patient_dir.name,
            'num_slices': len(slices_with_position),
            'pixel_spacing': [float(x) for x in first_slice.PixelSpacing],
            'slice_thickness': float(first_slice.SliceThickness),
            'z_spacing': z_spacing,
            'rescale_slope': float(first_slice.RescaleSlope),
            'rescale_intercept': float(first_slice.RescaleIntercept),
            'rows': int(first_slice.Rows),
            'columns': int(first_slice.Columns),
            'image_orientation': [float(x) for x in first_slice.ImageOrientationPatient],
            'z_positions': z_positions,
            'kvp': float(first_slice.KVP) if hasattr(first_slice, 'KVP') else None,
        }
        
        # Stack pixel arrays into 3D volume
        volume = np.stack([ds.pixel_array for _, ds in slices_with_position], axis=0)
        
        print(f"  Volume shape: {volume.shape}")
        print(f"  Pixel spacing: {metadata['pixel_spacing']} mm")
        print(f"  Slice thickness: {metadata['slice_thickness']} mm")
        print(f"  Z-spacing: {metadata['z_spacing']:.4f} mm")
        
        return volume, metadata
    
    def convert_to_hounsfield_units(self, volume: np.ndarray, metadata: Dict) -> np.ndarray:
        """
        Convert raw pixel values to Hounsfield Units
        HU = pixel_value * RescaleSlope + RescaleIntercept
        """
        slope = metadata['rescale_slope']
        intercept = metadata['rescale_intercept']
        
        hu_volume = volume.astype(np.float32) * slope + intercept
        
        # Validation check
        print(f"  HU range: {hu_volume.min():.1f} to {hu_volume.max():.1f}")
        print(f"  Expected: Air ~-1000, Water ~0, Bone >400")
        
        return hu_volume
    
    def apply_bone_windowing(self, hu_volume: np.ndarray) -> np.ndarray:
        """
        Apply bone windowing: Width=4000, Level=700
        Clips and normalizes to [0, 1] range
        """
        window_min = self.BONE_WINDOW_LEVEL - self.BONE_WINDOW_WIDTH / 2
        window_max = self.BONE_WINDOW_LEVEL + self.BONE_WINDOW_WIDTH / 2
        
        print(f"  Applying bone window: [{window_min:.0f}, {window_max:.0f}] HU")
        
        # Clip to window range
        windowed = np.clip(hu_volume, window_min, window_max)
        
        # Normalize to [0, 1]
        windowed = (windowed - window_min) / (window_max - window_min)
        
        return windowed.astype(np.float32)
    
    def reconstruct_coronal_view(self, axial_volume: np.ndarray, metadata: Dict) -> np.ndarray:
        """
        Reconstruct coronal view from axial slices using SimpleITK
        Target: isotropic spacing of 0.335mm (as per spec)
        
        Input: axial_volume shape (slices, height, width)
        Output: coronal_volume shape (slices, height, width)
        
        Note: This reconstructs from the FULL volume. Phase 2 will extract
        temporal bone ROIs from both axial and coronal views.
        """
        print(f"  Reconstructing coronal view...")
        
        # Convert numpy array to SimpleITK image
        sitk_image = sitk.GetImageFromArray(axial_volume)
        
        # Set original spacing [x, y, z]
        # DICOM PixelSpacing is [Row (Y), Column (X)]
        # SimpleITK expects [X, Y, Z]
        original_spacing = [
            metadata['pixel_spacing'][1],  # X spacing (Column)
            metadata['pixel_spacing'][0],  # Y spacing (Row)
            metadata['z_spacing']          # Z spacing (derived from positions)
        ]
        sitk_image.SetSpacing(original_spacing)
        
        print(f"  Original spacing: {original_spacing}")
        
        # Target isotropic spacing (0.335mm as per spec)
        target_spacing = [0.335, 0.335, 0.335]
        
        # Calculate new size to maintain physical dimensions
        original_size = sitk_image.GetSize()
        new_size = [
            int(round(original_size[0] * (original_spacing[0] / target_spacing[0]))),
            int(round(original_size[1] * (original_spacing[1] / target_spacing[1]))),
            int(round(original_size[2] * (original_spacing[2] / target_spacing[2])))
        ]
        
        print(f"  Resampling from {original_size} to {new_size}")
        
        # Resample to isotropic spacing
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(target_spacing)
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(sitk_image.GetDirection())
        resampler.SetOutputOrigin(sitk_image.GetOrigin())
        resampler.SetTransform(sitk.Transform())
        resampler.SetDefaultPixelValue(0)
        resampler.SetInterpolator(sitk.sitkLinear)
        
        resampled_image = resampler.Execute(sitk_image)
        
        # Permute axes to get coronal view
        # Axial: (Z, Y, X) -> Coronal: (Y, Z, X)
        # Note: SimpleITK Image (X, Y, Z) -> Permute [0, 2, 1] -> (X, Z, Y) -> Numpy (Y, Z, X)
        coronal_image = sitk.PermuteAxes(resampled_image, [0, 2, 1])
        
        # Convert back to numpy
        coronal_volume = sitk.GetArrayFromImage(coronal_image)
        
        print(f"  Coronal volume shape: {coronal_volume.shape}")
        
        return coronal_volume
    
    def split_lateral_hemispheres(self, volume: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split volume into left and right hemispheres with overlap
        
        DICOM convention: Right side of image = Left side of patient
        
        Returns:
            left_volume: Patient's left ear (right side of image)
            right_volume: Patient's right ear (left side of image)
        """
        _, _, width = volume.shape
        midline = width // 2
        
        print(f"  Splitting at midline {midline} with {self.MIDLINE_MARGIN}px margin")
        
        # Patient's RIGHT ear = LEFT side of image
        right_volume = volume[:, :, 0:midline + self.MIDLINE_MARGIN]
        
        # Patient's LEFT ear = RIGHT side of image
        left_volume = volume[:, :, midline - self.MIDLINE_MARGIN:]
        
        print(f"  Right ear volume: {right_volume.shape}")
        print(f"  Left ear volume: {left_volume.shape}")
        
        return left_volume, right_volume
    
    def process_patient(self, patient_id: str) -> bool:
        """Process a single patient through the full pipeline"""
        print(f"\n{'='*60}")
        print(f"Processing {patient_id}")
        print(f"{'='*60}")
        
        patient_dir = self.input_dir / patient_id
        
        if not patient_dir.exists():
            print(f"  ERROR: Directory not found: {patient_dir}")
            return False
        
        try:
            # Load and sort DICOM slices
            axial_volume, metadata = self.load_and_sort_dicom_series(patient_dir)
            
            # Convert to Hounsfield Units
            print(f"\n  Converting to Hounsfield Units...")
            hu_volume = self.convert_to_hounsfield_units(axial_volume, metadata)
            
            # Apply bone windowing
            print(f"\n  Applying bone windowing...")
            windowed_volume = self.apply_bone_windowing(hu_volume)
            
            # Reconstruct coronal view from FULL volume first
            print(f"\n  Reconstructing coronal view...")
            coronal_volume = self.reconstruct_coronal_view(windowed_volume, metadata)
            
            # Split left/right for both axial and coronal
            print(f"\n  Splitting lateral hemispheres (axial)...")
            left_axial, right_axial = self.split_lateral_hemispheres(windowed_volume)
            
            print(f"\n  Splitting lateral hemispheres (coronal)...")
            left_coronal, right_coronal = self.split_lateral_hemispheres(coronal_volume)
            
            # Save processed data
            print(f"\n  Saving processed data...")
            self.save_processed_data(
                patient_id, metadata,
                left_axial, right_axial,
                left_coronal, right_coronal
            )
            
            print(f"\n  Successfully processed {patient_id}")
            return True
            
        except Exception as e:
            print(f"\n  ERROR processing {patient_id}: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def save_processed_data(
        self,
        patient_id: str,
        metadata: Dict,
        left_axial: np.ndarray,
        right_axial: np.ndarray,
        left_coronal: np.ndarray,
        right_coronal: np.ndarray
    ):
        """Save processed volumes and metadata"""
        
        # Create output directories
        patient_output = self.output_dir / patient_id
        left_dir = patient_output / 'left'
        right_dir = patient_output / 'right'
        
        left_dir.mkdir(parents=True, exist_ok=True)
        right_dir.mkdir(parents=True, exist_ok=True)
        
        # Save left ear
        np.save(left_dir / 'axial_volume.npy', left_axial)
        np.save(left_dir / 'coronal_volume.npy', left_coronal)
        
        left_metadata = metadata.copy()
        left_metadata['side'] = 'L'
        left_metadata['axial_shape'] = list(left_axial.shape)
        left_metadata['coronal_shape'] = list(left_coronal.shape)
        
        with open(left_dir / 'metadata.json', 'w') as f:
            json.dump(left_metadata, f, indent=2)
        
        # Save right ear
        np.save(right_dir / 'axial_volume.npy', right_axial)
        np.save(right_dir / 'coronal_volume.npy', right_coronal)
        
        right_metadata = metadata.copy()
        right_metadata['side'] = 'R'
        right_metadata['axial_shape'] = list(right_axial.shape)
        right_metadata['coronal_shape'] = list(right_coronal.shape)
        
        with open(right_dir / 'metadata.json', 'w') as f:
            json.dump(right_metadata, f, indent=2)
        
        print(f"    Saved to {patient_output}")
    
    def process_all_patients(self):
        """Process all patients in the input directory"""
        
        # Get list of patient directories
        patient_dirs = sorted([d for d in self.input_dir.iterdir() if d.is_dir()])
        
        print(f"\nFound {len(patient_dirs)} patient directories")
        print(f"Output directory: {self.output_dir}")
        
        # Process each patient
        results = {}
        for patient_dir in patient_dirs:
            patient_id = patient_dir.name
            success = self.process_patient(patient_id)
            results[patient_id] = success
        
        # Summary
        print(f"\n{'='*60}")
        print(f"PROCESSING SUMMARY")
        print(f"{'='*60}")
        
        successful = sum(results.values())
        total = len(results)
        
        print(f"\nSuccessfully processed: {successful}/{total} patients")
        
        if successful < total:
            print(f"\nFailed patients:")
            for patient_id, success in results.items():
                if not success:
                    print(f"  - {patient_id}")
        
        return results
