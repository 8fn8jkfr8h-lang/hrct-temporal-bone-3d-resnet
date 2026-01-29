"""
Unified ROI Viewer & Editor for Temporal Bone HRCT
Combines visualization and manual annotation capabilities.

Usage:
    python -m utils.viewer_roi

Controls:
    [Navigation]
    Arrows    : Navigate slices
    a / c     : Switch Axial / Coronal Views
    l / r     : Switch Left / Right Ear
    n / p     : Next / Prev Patient
    q         : Quit

    [View Mode]
    o         : Toggle Overlays (ROI + Landmarks)
    Home/End  : Jump to ROI Start/End

    [Edit Mode]
    e         : Toggle EDIT MODE (Axial View Only)
    Click     : Set ROI Center (X, Y)
    [         : Set Inferior Z Limit (Bottom)
    ]         : Set Superior Z Limit (Top)
    s         : Save ROI to disk
    Esc       : Discard changes / Reload
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
from pathlib import Path
import sys

# Inherit from the robust base viewer
# Ensure utils is in path if running as script
if __name__ == '__main__' and __package__ is None:
    sys.path.append(str(Path(__file__).parent.parent))

from utils.viewer_interactive import InteractiveViewer

class ROIEditor(InteractiveViewer):
    """
    Combined Viewer and Editor for ROIs.
    Inherits data loading and display logic from InteractiveViewer.
    Adds overlay rendering and editing interactions.
    """
    
    ROI_COLOR = '#00FF00'     # Green for saved
    ROI_EDIT_COLOR = '#00FFFF' # Cyan for unsaved edits
    LANDMARK_COLORS = {'apex': 'red', 'basal': 'green', 'round_window': 'blue'}
    
    def __init__(self, processed_dir='processed_data', roi_dir='roi_extracted', landmarks_dir='landmarks_detected'):
        # Directories
        self.processed_dir = Path(processed_dir)
        self.roi_dir = Path(roi_dir)
        self.landmarks_dir = Path(landmarks_dir)
        
        # Ensure ROI directory exists
        self.roi_dir.mkdir(parents=True, exist_ok=True)
        
        # State
        self.edit_mode = False
        self.show_overlay = True
        self.unsaved_changes = False
        
        # Current ROI Data
        self.roi_data = {
            'center': None, # (z, y, x)
            'z_min': None,
            'z_max': None,
            'loaded': False
        }
        self.landmarks = {}
        
        # Visual Artists
        self.overlay_artists = []
        self.status_box = None
        self.mode_text = None
        
        # Initialize Base Viewer
        # This will call load_current_data() once
        super().__init__(data_dir=processed_dir)
        
        # Final Setup
        self._setup_extra_ui()
        self.update_status_overlay()

    def setup_figure(self):
        """Override to add click handlers"""
        super().setup_figure()
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

    def _setup_extra_ui(self):
        """Add persistent UI elements"""
        # Status Box (Top Right)
        self.status_box = self.fig.text(0.99, 0.95, '', fontsize=9, 
                                       color='white', verticalalignment='top', 
                                       horizontalalignment='right',
                                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
        
        # Mode Indicator (Top Left, under coords)
        self.mode_text = self.fig.text(0.01, 0.94, 'VIEW MODE', fontsize=12, 
                                      color='lime', fontweight='bold',
                                      bbox=dict(facecolor='black', alpha=0.7))

        # Controls Help (Bottom Left - Overwrite base)
        help_text = (
            "CONTROLS:\n"
            "e       : Toggle EDIT Mode (Axial Only)\n"
            "Click   : Set ROI Center (Auto-sets Z-bounds)\n"
            "w       : WRITE (Save) ROI to disk\n"
            "o       : Toggle Overlay (ROI + Landmarks)\n"
            "Arrows  : Navigate Slices\n"
            "a / c   : Switch Axial/Coronal View\n"
            "l / r   : Switch Left/Right Ear\n"
            "n / p   : Next/Prev Patient\n"
            "Esc     : Reload/Discard Changes"
        )
        self.fig.text(0.01, 0.01, help_text, fontsize=9, color='white', va='bottom',
                      bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    def load_current_data(self):
        """
        Extended data loader.
        1. Calls base loader (Volumes).
        2. Loads ROI metadata.
        3. Loads Landmarks.
        """
        super().load_current_data()
        
        # Reset State
        self.roi_data = {'center': None, 'z_min': None, 'z_max': None, 'loaded': False}
        self.landmarks = {}
        self.unsaved_changes = False
        
        if self.axial is None:
            return # No volume data

        patient_id = self.patients[self.patient_idx].name
        
        # 1. Load ROI
        roi_path = self.roi_dir / patient_id / self.side / 'roi_metadata.json'
        if roi_path.exists():
            try:
                with open(roi_path, 'r') as f:
                    meta = json.load(f)
                    c = meta['center_xyz_voxel']
                    z_bounds = meta['bounds']['z']
                    self.roi_data = {
                        'center': tuple(c), # (z, y, x)
                        'z_min': z_bounds[0],
                        'z_max': z_bounds[1],
                        'loaded': True
                    }
            except Exception as e:
                print(f"Error loading ROI: {e}")
        
        # 2. Load Landmarks
        lm_path = self.landmarks_dir / patient_id / self.side / 'landmark_coords.json'
        if lm_path.exists():
            try:
                with open(lm_path, 'r') as f:
                    self.landmarks = json.load(f).get('landmarks', {})
            except Exception as e:
                print(f"Error loading landmarks: {e}")

        # Refresh UI
        if hasattr(self, 'status_box'):
            self.update_status_overlay()

    def save_roi(self):
        """Save current ROI configuration to disk"""
        if not self.roi_data['center'] or self.roi_data['z_min'] is None:
            print("Cannot save: Incomplete ROI definition.")
            return

        c_z, c_y, c_x = self.roi_data['center']
        z_min, z_max = self.roi_data['z_min'], self.roi_data['z_max']
        
        # Validate
        if z_min >= z_max:
            print("Invalid Z-Range: Min >= Max")
            return
            
        # 1. Extract Volume
        roi_size = 128 # Fixed size (128x128x128)
        half = roi_size // 2
        
        # Coordinates
        y_min, y_max = c_y - half, c_y + half
        x_min, x_max = c_x - half, c_x + half
        
        # Handle Padding
        vol_z, vol_y, vol_x = self.axial.shape
        
        # Extract Logic (with padding if needed)
        # Simplified: Slice and Pad
        # Initialize canvas
        roi_vol = np.zeros((z_max - z_min, roi_size, roi_size), dtype=self.axial.dtype)
        
        # Iterate slices to copy (safer than complex slicing logic for edges)
        for i, z in enumerate(range(z_min, z_max)):
            if 0 <= z < vol_z:
                # Valid Z
                # Calculate source/dest ranges
                # Y-dimension
                src_y_start = max(0, y_min)
                src_y_end = min(vol_y, y_max)
                dst_y_start = max(0, -y_min)
                dst_y_end = dst_y_start + (src_y_end - src_y_start)
                
                # X-dimension
                src_x_start = max(0, x_min)
                src_x_end = min(vol_x, x_max)
                dst_x_start = max(0, -x_min)
                dst_x_end = dst_x_start + (src_x_end - src_x_start)
                
                if (src_y_end > src_y_start) and (src_x_end > src_x_start):
                    roi_vol[i, dst_y_start:dst_y_end, dst_x_start:dst_x_end] = \
                        self.axial[z, src_y_start:src_y_end, src_x_start:src_x_end]

        # 2. Save File
        patient_id = self.patients[self.patient_idx].name
        out_dir = self.roi_dir / patient_id / self.side
        out_dir.mkdir(parents=True, exist_ok=True)
        
        np.save(out_dir / 'axial_roi.npy', roi_vol)
        
        # 3. Save Metadata
        meta = {
            'patient_id': patient_id,
            'side': self.side,
            'center_xyz_voxel': [int(c_z), int(c_y), int(c_x)],
            'bounds': {
                'z': (int(z_min), int(z_max)),
                'y': (int(y_min), int(y_max)),
                'x': (int(x_min), int(x_max))
            },
            'method': 'manual_human_in_loop',
            'roi_shape': roi_vol.shape,
            'qc_passed': True,
            'qc_message': 'Manual update via viewer'
        }
        with open(out_dir / 'roi_metadata.json', 'w') as f:
            json.dump(meta, f, indent=2)
            
        # 4. Generate Preview Image (consistent with Phase 2b)
        try:
            mid_z = roi_vol.shape[0] // 2
            plt.imsave(out_dir / 'roi_preview.png', roi_vol[mid_z], cmap='gray')
        except Exception as e:
            print(f"Warning: Could not save preview image: {e}")

        print(f"✅ ROI Saved for {patient_id} {self.side}")
        self.unsaved_changes = False
        self.roi_data['loaded'] = True
        self.update_status_overlay()
        self.update_display()

    def on_click(self, event):
        """Handle mouse clicks for editing"""
        if not self.edit_mode: return
        if event.inaxes != self.ax: return
        if self.view != 'axial': return # Restrict editing to Axial
        
        if event.button == 1: # Left Click
            # Update Center
            x, y = int(event.xdata), int(event.ydata)
            z = self.slice_idx
            
            # Automatically set Z-bounds based on fixed depth of 128
            roi_depth = 128
            half_depth = roi_depth // 2
            z_min = z - half_depth
            z_max = z + half_depth
            
            self.roi_data['center'] = (z, y, x)
            self.roi_data['z_min'] = z_min
            self.roi_data['z_max'] = z_max
            
            self.unsaved_changes = True
            self.update_display()
            self.update_status_overlay()

    def on_key(self, event):
        """Handle key presses"""
        # Global Keys
        if event.key == 'e':
            self.edit_mode = not self.edit_mode
            self.mode_text.set_text("EDIT MODE" if self.edit_mode else "VIEW MODE")
            self.mode_text.set_color('red' if self.edit_mode else 'lime')
            
            # If entering edit mode and no ROI, init center at current slice center
            if self.edit_mode and self.roi_data['center'] is None and self.axial is not None:
                d, h, w = self.axial.shape
                z = self.slice_idx
                roi_depth = 128
                half_depth = roi_depth // 2
                
                self.roi_data['center'] = (z, h//2, w//2)
                self.roi_data['z_min'] = z - half_depth
                self.roi_data['z_max'] = z + half_depth
                self.unsaved_changes = True
            
            self.update_display()
            return

        if event.key == 'o':
            self.show_overlay = not self.show_overlay
            self.update_display()
            return
            
        if event.key == 'w' and self.edit_mode:
            self.save_roi()
            return
            
        if event.key == 'escape':
            self.load_current_data() # Reload from disk
            self.update_display()
            return

        # ROI Navigation
        if event.key == 'home' and self.roi_data['z_min'] is not None:
            if self.view == 'axial':
                self.slice_idx = self.roi_data['z_min']
                self.update_display()
            return
        elif event.key == 'end' and self.roi_data['z_max'] is not None:
            if self.view == 'axial':
                self.slice_idx = self.roi_data['z_max']
                self.update_display()
            return

        # ROI Navigation
        if event.key == 'home' and self.roi_data['z_min'] is not None:
            if self.view == 'axial':
                self.slice_idx = self.roi_data['z_min']
                self.update_display()
            return
        elif event.key == 'end' and self.roi_data['z_max'] is not None:
            if self.view == 'axial':
                self.slice_idx = self.roi_data['z_max']
                self.update_display()
            return

        # Pass to base class for navigation
        super().on_key(event)

    def update_display(self):
        """Render the volume (base) then draw overlays"""
        # 1. Draw Base Volume
        super().update_display()
        
        # 2. Cleanup Old Overlays
        for artist in self.overlay_artists:
            artist.remove()
        self.overlay_artists = []
        
        # Add Legend if not present
        if not hasattr(self, 'legend_added'):
            self._draw_legend()
            self.legend_added = True

        if not self.show_overlay: return
        if self.axial is None: return
        
        # --- Calculate Scaling Factors for Coronal View ---
        # Phase 1 resamples Coronal to 0.335mm isotropic.
        # Axial coordinates must be scaled to match this new resolution.
        target_spacing = 0.335
        try:
            # Metadata keys might vary slightly, handle robustness
            z_spacing = self.metadata.get('z_spacing', self.metadata.get('slice_thickness', 0.6))
            
            # Pixel spacing is [row, col] -> [y, x]
            p_spacing = self.metadata.get('pixel_spacing', [0.3, 0.3])
            if isinstance(p_spacing, (int, float)): p_spacing = [p_spacing, p_spacing]
            y_spacing, x_spacing = p_spacing[0], p_spacing[1]
            
            scale_z = z_spacing / target_spacing
            scale_y = y_spacing / target_spacing
            scale_x = x_spacing / target_spacing
        except Exception:
            # Fallback if metadata missing
            scale_z, scale_y, scale_x = 1.0, 1.0, 1.0

        # 3. Draw ROI Box
        # Logic depends on View (Axial vs Coronal)
        
        if self.roi_data['center']:
            c_z, c_y, c_x = self.roi_data['center']
            roi_size = 128
            half = roi_size // 2
            
            # Determine Color
            color = self.ROI_EDIT_COLOR if self.unsaved_changes else self.ROI_COLOR
            style = '--' if self.unsaved_changes else '-'
            
            if self.view == 'axial':
                # --- AXIAL VIEW ---
                # Rect is (X, Y)
                rect_x = c_x - half
                rect_y = c_y - half
                
                is_center = (self.slice_idx == c_z)
                in_bounds = False
                if self.roi_data['z_min'] is not None and self.roi_data['z_max'] is not None:
                    if self.roi_data['z_min'] <= self.slice_idx < self.roi_data['z_max']:
                        in_bounds = True
                
                if is_center or in_bounds or self.edit_mode:
                    alpha = 1.0 if is_center else 0.5
                    linewidth = 2 if is_center else 1
                    
                    rect = patches.Rectangle((rect_x, rect_y), roi_size, roi_size,
                                             linewidth=linewidth, edgecolor=color,
                                             facecolor='none', linestyle=style, alpha=alpha)
                    self.ax.add_patch(rect)
                    self.overlay_artists.append(rect)
                    
            elif self.view == 'coronal':
                # --- CORONAL VIEW ---
                # 1. Coordinate Transform: Axial Voxel -> Scaled Isotropic Voxel
                # Coronal Volume is (Y_iso, Z_iso, X_iso)
                
                # Transform Bounds to Scaled Space
                x_min_true = (c_x - half) * scale_x
                x_max_true = (c_x + half) * scale_x
                z_min_true = self.roi_data['z_min'] * scale_z if self.roi_data['z_min'] else 0
                z_max_true = self.roi_data['z_max'] * scale_z if self.roi_data['z_max'] else 0
                
                # Check Y-bounds (Slice check)
                # self.slice_idx is index in Coronal Array (Y_iso)
                # ROI center Y is c_y (Axial Voxel)
                y_min_iso = (c_y - half) * scale_y
                y_max_iso = (c_y + half) * scale_y
                
                if y_min_iso <= self.slice_idx < y_max_iso:
                    
                    if self.roi_data['z_min'] is not None and self.roi_data['z_max'] is not None:
                        # 2. View Transform: Flips
                        # Coronal View flips Z (Height) and X (Width)
                        # Display Width/Height from the actual Coronal Volume shape
                        if self.coronal is not None:
                            # self.coronal is (Y, Z, X)
                            height = self.coronal.shape[1] # Z_iso
                            width = self.coronal.shape[2]  # X_iso
                            
                            # Flip X: Display X = Width - 1 - X_iso
                            # Left Display Edge corresponds to Max X_iso
                            disp_x_left = width - 1 - x_max_true
                            disp_width = x_max_true - x_min_true
                            
                            # Flip Z: Display Y = Height - 1 - Z_iso
                            # Top Display Edge corresponds to Max Z_iso
                            disp_y_top = height - 1 - z_max_true
                            disp_height = z_max_true - z_min_true
                            
                            rect = patches.Rectangle((disp_x_left, disp_y_top), disp_width, disp_height,
                                                     linewidth=1, edgecolor=color,
                                                     facecolor='none', linestyle=style, alpha=0.6)
                            self.ax.add_patch(rect)
                            self.overlay_artists.append(rect)

        # 4. Draw Landmarks
        if self.landmarks:
            markers = {'apex': 'o', 'basal': 's', 'round_window': '^'}
            
            for name, coord in self.landmarks.items():
                if coord is None: continue
                lz, ly, lx = coord
                
                marker = markers.get(name, 'o')
                color = self.LANDMARK_COLORS.get(name, 'yellow')
                
                if self.view == 'axial':
                    if abs(lz - self.slice_idx) < 3:
                        sc = self.ax.scatter([lx], [ly], c=color, marker=marker, 
                                            s=20, edgecolors='white', linewidths=0.5)
                        self.overlay_artists.append(sc)
                
                elif self.view == 'coronal':
                    # Scale Y to check visibility
                    ly_iso = ly * scale_y
                    
                    if abs(ly_iso - self.slice_idx) < 3:
                        # Scale X and Z
                        lx_iso = lx * scale_x
                        lz_iso = lz * scale_z
                        
                        if self.coronal is not None:
                            width = self.coronal.shape[2]
                            height = self.coronal.shape[1]
                            
                            dx = width - 1 - lx_iso
                            dy = height - 1 - lz_iso
                            
                            sc = self.ax.scatter([dx], [dy], c=color, marker=marker,
                                                s=20, edgecolors='white', linewidths=0.5)
                            self.overlay_artists.append(sc)

    def _draw_legend(self):
        """Draw persistent legend"""
        legend_elements = [
            patches.Patch(facecolor=self.ROI_COLOR, alpha=0.3, edgecolor=self.ROI_COLOR, label='ROI'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=6, label='Apex'),
            plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='green', markersize=6, label='Basal'),
            plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='blue', markersize=6, label='RW'),
        ]
        self.fig.legend(handles=legend_elements, loc='lower right', fontsize=8, framealpha=0.6)

    def update_status_overlay(self):
        """Update the text box with current ROI stats"""
        if not hasattr(self, 'status_box'): return
        
        patient_id = self.patients[self.patient_idx].name
        status = f"{patient_id} ({self.side.upper()})\n"
        status += "-" * 20 + "\n"
        
        # ROI Status
        if self.roi_data['center']:
            c_z, c_y, c_x = self.roi_data['center']
            status += f"Center: {int(c_x)}, {int(c_y)}, {int(c_z)}\n"
        else:
            status += "Center: NOT SET\n"
            
        z_min = self.roi_data['z_min']
        z_max = self.roi_data['z_max']
        
        zm_str = f"{int(z_min)}" if z_min is not None else "N/A"
        zM_str = f"{int(z_max)}" if z_max is not None else "N/A"
        
        status += f"Z-Range: {zm_str} - {zM_str}\n"
        
        if z_min is not None and z_max is not None:
            depth = int(z_max - z_min)
            status += f"Depth: {depth} slices\n"
        
        # State Indicators
        if self.unsaved_changes:
            status += "\n⚠️ UNSAVED CHANGES"
            self.status_box.set_bbox(dict(facecolor='darkred', alpha=0.5, boxstyle='round'))
        elif self.roi_data['loaded']:
            status += "\n✅ ROI SAVED"
            self.status_box.set_bbox(dict(facecolor='darkgreen', alpha=0.5, boxstyle='round'))
        else:
            status += "\n⚪ NO ROI DATA"
            self.status_box.set_bbox(dict(facecolor='black', alpha=0.5, boxstyle='round'))
            
        self.status_box.set_text(status)
        self.fig.canvas.draw_idle()

if __name__ == '__main__':
    print("Launching ROI Editor...")
    editor = ROIEditor()
    editor.show()