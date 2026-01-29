"""
Phase 2B: Middle Ear ROI Extraction from Landmarks

This script takes the landmarks detected in Phase 2A and extracts the
final Region of Interest (ROI) for the classification model.

Process:
1. Load landmark coordinates (Apex, Basal, Round Window).
2. Compute Middle Ear Center using anatomical offsets from the Basal turn.
3. Determine Z-axis bounds (superior/inferior limits).
4. Extract the 3D ROI (Fixed X-Y size, Variable Z size).
5. Perform Quality Control (QC) checks.
6. Save ROI and visualization.
"""

import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

class ROIExtractor:
    def __init__(self, 
                 processed_dir='processed_data',
                 landmarks_dir='landmarks_detected', 
                 output_dir='roi_extracted'):
        self.processed_dir = Path(processed_dir)
        self.landmarks_dir = Path(landmarks_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # ROI parameters (matching project specs)
        self.roi_xy_size = 128  # pixels (approx 42.88mm)
        self.roi_z_size = 128   # Fixed Z-depth (approx 42.88mm)
        
        # Anatomical offsets (in mm)
        # HRCT spacing is isotropic 0.335 mm
        self.spacing_mm = 0.335
        
        # Offsets from the Basal Turn to the Middle Ear Center
        # These values are initial estimates and should be calibrated.
        self.offsets_mm = {
            'lateral': 6.0,    # Middle ear is ~6mm lateral to cochlea
            # 'superior'/'inferior' offsets no longer used for dynamic sizing
            # but 'center' calculation might still need adjustment if desired.
            # For now, we center Z on the Basal Turn (offset=0 implied).
        }
        
    def compute_middle_ear_center(self, landmarks, side):
        """
        Compute middle ear center from cochlear landmarks.
        
        Strategy:
        - Z (Vertical): Basal turn level.
        - Y (Anterior-Posterior): Average of Apex and Basal Y.
        - X (Lateral): Lateral to Basal turn. Direction depends on ear side.
        
        Args:
            landmarks: dict with apex, basal, round_window coordinates (z, y, x)
            side: 'left' or 'right'
            
        Returns:
            center: (z, y, x) in voxel coordinates
        """
        apex = np.array(landmarks['apex']) if landmarks.get('apex') else None
        basal = np.array(landmarks['basal']) if landmarks.get('basal') else None
        
        # Fallback Logic
        if basal is None and apex is None:
            raise ValueError("Critical failure: Both Apex and Basal landmarks are missing.")

        if basal is None:
            # Estimate Basal from Apex
            # Basal is roughly 4.5mm inferior to Apex (approx 13 slices at 0.335mm)
            z_offset_px = 4.5 / self.spacing_mm
            print(f"   ⚠️  Missing Basal: Estimating from Apex (Offset Z +{z_offset_px:.1f}px)")
            basal = apex + np.array([z_offset_px, 0, 0]) # Crude approximation
            
        # 1. Z Center: Basal turn level
        z_center = basal[0]
        
        # 2. Y Center: Average of Apex and Basal Y
        if apex is not None:
            y_center = (apex[1] + basal[1]) / 2
        else:
            print("   ⚠️  Missing Apex: Using Basal Y for center")
            y_center = basal[1]
        
        # 3. X Center: Middle ear is LATERAL to the cochlea
        lateral_offset_px = self.offsets_mm['lateral'] / self.spacing_mm
        
        if side == 'left':
            # Moving laterally means increasing X index
            x_center = basal[2] + lateral_offset_px
        else: # right
            # Moving laterally means decreasing X index
            x_center = basal[2] - lateral_offset_px
            
        return np.array([z_center, y_center, x_center])
    
    def compute_z_bounds(self, center, volume_shape):
        """
        Compute fixed Z bounds based on center.
        """
        z_center = int(center[0])
        half_z = self.roi_z_size // 2
        
        z_min = z_center - half_z
        z_max = z_center + half_z
        
        # Return unclipped bounds to allow padding calculation in extract_roi
        return z_min, z_max
    
    def extract_roi(self, volume, center, z_min, z_max):
        """
        Extract the ROI volume with fixed size, padding if necessary.
        """
        # Center is (z, y, x)
        _, y_center, x_center = center
        
        # XY Bounds
        half = self.roi_xy_size // 2
        y_min = int(y_center - half)
        y_max = int(y_center + half)
        x_min = int(x_center - half)
        x_max = int(x_center + half)
        
        # Calculate Padding
        # Z Padding
        pad_z_before = max(0, -z_min)
        pad_z_after = max(0, z_max - volume.shape[0])
        
        # XY Padding
        pad_y_before = max(0, -y_min)
        pad_y_after = max(0, y_max - volume.shape[1])
        pad_x_before = max(0, -x_min)
        pad_x_after = max(0, x_max - volume.shape[2])
        
        # Clamp extraction coords to valid volume indices
        z_min_c = max(0, z_min)
        z_max_c = min(volume.shape[0], z_max)
        y_min_c = max(0, y_min)
        y_max_c = min(volume.shape[1], y_max)
        x_min_c = max(0, x_min)
        x_max_c = min(volume.shape[2], x_max)
        
        # Extract available data
        roi_slice = volume[z_min_c:z_max_c, y_min_c:y_max_c, x_min_c:x_max_c]
        
        # Pad to maintain fixed size (128, 128, 128)
        if (pad_z_before > 0 or pad_z_after > 0 or 
            pad_y_before > 0 or pad_y_after > 0 or 
            pad_x_before > 0 or pad_x_after > 0):
            
            roi_slice = np.pad(roi_slice, 
                               ((pad_z_before, pad_z_after), 
                                (pad_y_before, pad_y_after), 
                                (pad_x_before, pad_x_after)),
                               mode='constant', constant_values=0)
            
        bounds = {
            'x': (x_min, x_max),
            'y': (y_min, y_max),
            'z': (z_min, z_max)
        }
        
        return roi_slice, bounds

    def qc_check(self, roi):
        """
        Basic Quality Control
        """
        # 1. Dimensions
        if roi.shape != (self.roi_z_size, self.roi_xy_size, self.roi_xy_size):
             return False, f"Invalid dimensions: {roi.shape}"

        # 2. Content (Bone Fraction)
        # Bone is high intensity. In 0-1 scale, bone is usually > 0.3 or 0.4
        bone_mask = roi > 0.3
        bone_fraction = np.sum(bone_mask) / roi.size
        
        if bone_fraction < 0.05:
            return False, f"Low bone content: {bone_fraction:.3f}"
            
        return True, f"OK (Bone: {bone_fraction:.2f})"

    def visualize(self, roi, output_path):
        """Save a middle slice visualization"""
        mid_z = roi.shape[0] // 2
        plt.figure(figsize=(6, 6))
        plt.imshow(roi[mid_z], cmap='gray')
        plt.title(f"ROI Middle Slice (Z={mid_z})")
        plt.axis('off')
        plt.savefig(output_path)
        plt.close()

    def process_all(self):
        # Find all detected landmarks
        landmark_files = sorted(self.landmarks_dir.glob('*/*/landmark_coords.json'))
        
        if not landmark_files:
            print("No landmarks found. Run Phase 2A first.")
            return

        results = []
        
        for lf in tqdm(landmark_files, desc="Extracting ROIs"):
            patient_id = lf.parent.parent.name
            side = lf.parent.name
            
            # Load Landmarks
            with open(lf) as f:
                data = json.load(f)
                
            landmarks = data.get('landmarks')
            if not landmarks:
                print(f"Skipping {patient_id} {side}: No landmark data found")
                continue
                
            # Load Volume
            vol_path = self.processed_dir / patient_id / side / 'axial_volume.npy'
            if not vol_path.exists(): continue
            volume = np.load(vol_path)
            
            try:
                # Compute Center & Bounds
                center = self.compute_middle_ear_center(landmarks, side)
                z_min, z_max = self.compute_z_bounds(center, volume.shape)
                
                # Extract
                roi, bounds = self.extract_roi(volume, center, z_min, z_max)
                
                # QC
                passed, msg = self.qc_check(roi)
                
                # Save
                out_dir = self.output_dir / patient_id / side
                out_dir.mkdir(parents=True, exist_ok=True)
                
                np.save(out_dir / 'axial_roi.npy', roi)
                
                meta = {
                    'patient_id': patient_id,
                    'side': side,
                    'center_xyz_voxel': center.tolist(),
                    'bounds': bounds,
                    'qc_passed': passed,
                    'qc_message': msg,
                    'roi_shape': roi.shape
                }
                
                with open(out_dir / 'roi_metadata.json', 'w') as f:
                    json.dump(meta, f, indent=2)
                    
                self.visualize(roi, out_dir / 'roi_preview.png')
                
                results.append(meta)
            
            except ValueError as e:
                print(f"❌ Failed {patient_id} {side}: {e}")
            except Exception as e:
                print(f"❌ Error {patient_id} {side}: {e}")
            
        # Summary
        print(f"Processed {len(results)} ROIs.")

if __name__ == '__main__':
    extractor = ROIExtractor()
    extractor.process_all()
