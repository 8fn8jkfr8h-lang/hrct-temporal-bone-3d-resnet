"""
Interactive Viewer for Processed DICOM Data
Allows manual verification of preprocessing quality

Usage:
    python -m utils.viewer_interactive
    
    Or from Python:
    from utils.viewer_interactive import InteractiveViewer
    viewer = InteractiveViewer()
    viewer.show()

Controls:
- Arrow keys: Navigate through slices
- Mouse scroll: Navigate through slices
- 'a'/'c': Switch between axial and coronal views
- 'l'/'r': Switch between left and right ears
- 'n'/'p': Next/previous patient
- 'q': Quit
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import sys


class InteractiveViewer:
    """Interactive viewer for temporal bone CT data"""
    
    def __init__(self, data_dir='processed_data'):
        self.data_dir = Path(data_dir)
        self.patients = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        
        if len(self.patients) == 0:
            print(f"No patients found in {data_dir}")
            sys.exit(1)
        
        self.patient_idx = 0
        self.side = 'left'
        self.view = 'axial'
        self.slice_idx = 0
        
        # For smooth rendering
        self.image_handle = None
        self.title_handle = None
        
        self.load_current_data()
        self.setup_figure()
        
    def load_current_data(self):
        """Load data for current patient and side"""
        patient_id = self.patients[self.patient_idx].name
        side_dir = self.patients[self.patient_idx] / self.side
        
        # Initialize defaults
        self.axial = None
        self.coronal = None
        self.metadata = None
        
        # Check if data exists
        if not side_dir.exists():
            print(f"\nLoaded: {patient_id} - {self.side.upper()} ear (NOT FOUND / EXCLUDED)")
            
            # Attempt to load metadata from the OTHER side to show something useful
            other_side = 'right' if self.side == 'left' else 'left'
            other_side_dir = self.patients[self.patient_idx] / other_side
            
            if other_side_dir.exists():
                try:
                    with open(other_side_dir / 'metadata.json', 'r') as f:
                        # Load metadata and explicitly mark as inferred/fallback
                        self.metadata = json.load(f)
                        self.metadata['side'] = self.side.upper() + " (Inferred from " + other_side.upper() + ")"
                        print(f"  Loaded metadata from contralateral ({other_side}) ear.")
                except Exception:
                    pass
            
            self.slice_idx = 0
            return

        # Load volumes
        try:
            self.axial = np.load(side_dir / 'axial_volume.npy')
            self.coronal = np.load(side_dir / 'coronal_volume.npy')
            
            # Flip coronal view for proper radiological display
            self.coronal = np.flip(self.coronal, axis=1)  # Vertical flip for head up
            self.coronal = np.flip(self.coronal, axis=2)  # Horizontal flip for L/R convention
            
            # Load metadata
            with open(side_dir / 'metadata.json', 'r') as f:
                self.metadata = json.load(f)
            
            # Reset slice index to middle
            self.slice_idx = self.axial.shape[0] // 2
            
            print(f"\nLoaded: {patient_id} - {self.side.upper()} ear")
            print(f"Axial shape: {self.axial.shape}")
            print(f"Coronal shape: {self.coronal.shape}")
            
        except Exception as e:
            print(f"Error loading data: {e}")
            self.axial = None
            self.coronal = None
            self.metadata = None

    def get_current_volume(self):
        """Get currently selected volume"""
        if self.axial is None:
            return None
        return self.axial if self.view == 'axial' else self.coronal
    
    def setup_figure(self):
        """Setup matplotlib figure with optimized rendering"""
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        
        # Text for coordinate display
        self.coord_text = self.ax.text(0.01, 0.99, '', transform=self.ax.transAxes, 
                                      color='yellow', fontsize=10, verticalalignment='top',
                                      bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))
        
        # Initial display
        volume = self.get_current_volume()
        
        if volume is not None:
            self.image_handle = self.ax.imshow(volume[self.slice_idx], cmap='bone', 
                                               vmin=0, vmax=1, interpolation='bilinear')
        else:
            self.image_handle = None
            self.ax.text(0.5, 0.5, "Data not found / Excluded", 
                         ha='center', va='center', fontsize=14, color='red')
            
        self.ax.axis('off')
        
        # Title
        self.title_handle = self.ax.set_title('', fontsize=12, fontweight='bold', pad=20)
        
        # Controls text
        controls = "Controls: ←/→ slices | Scroll wheel | a/c axial/coronal | l/r left/right | n/p next/prev patient | q quit"
        self.fig.text(0.5, 0.02, controls, ha='center', fontsize=10, 
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        self.update_display()

    def on_mouse_move(self, event):
        """Handle mouse movement to display coordinates"""
        volume = self.get_current_volume()
        if event.inaxes == self.ax and volume is not None:
            x_disp, y_disp = int(event.xdata), int(event.ydata)
            
            # Calculate true volume coordinates based on view
            if self.view == 'axial':
                # Axial: (Z, Y, X)
                # Display X = Vol X
                # Display Y = Vol Y
                # Slice = Vol Z
                vol_x = x_disp
                vol_y = y_disp
                vol_z = self.slice_idx
                
            elif self.view == 'coronal':
                # Coronal: (Y, Z, X) but flipped
                # self.coronal = np.flip(self.coronal, axis=1)  # Flip Z (height)
                # self.coronal = np.flip(self.coronal, axis=2)  # Flip X (width)
                
                # Un-flip X
                width = volume.shape[2]
                vol_x = width - 1 - x_disp
                
                # Un-flip Z (Display Y maps to Volume Z)
                height = volume.shape[1]
                vol_z = height - 1 - y_disp
                
                # Slice = Vol Y
                vol_y = self.slice_idx
            
            self.coord_text.set_text(f"X: {vol_x}, Y: {vol_y}, Z: {vol_z}")
            self.fig.canvas.draw_idle()
        else:
            self.coord_text.set_text("")
            self.fig.canvas.draw_idle()
        
    def update_display(self):
        """Update the display (optimized for speed)"""
        volume = self.get_current_volume()
        patient_id = self.patients[self.patient_idx].name
        
        if volume is None:
            # Clear everything and show text
            self.ax.clear()
            self.ax.axis('off')
            
            # Show "No Data" text
            self.ax.text(0.5, 0.5, "Data not found / Excluded", 
                         ha='center', va='center', fontsize=14, color='red')
            
            # Re-create title handle since clear() removed it
            # Build title string
            title = f"{patient_id} - {self.side.upper()} ear - NO DATA / EXCLUDED\n"
            
            if self.metadata:
                side_info = self.metadata.get('side', 'other')
                if len(side_info) > 20: # Truncate if too long (e.g. inferred msg)
                    side_info = "contralateral"
                
                title += f"Metadata (from {side_info}):\n"
                title += f"Pixel spacing: {self.metadata.get('pixel_spacing', 'N/A')} mm"
                title += f" | Slice thickness: {self.metadata.get('slice_thickness', 'N/A')} mm"
            
            self.title_handle = self.ax.set_title(title, fontsize=12, fontweight='bold', pad=20)
            
            # Invalidate image handle so we know to re-create it next time
            self.image_handle = None
            
            self.fig.canvas.draw_idle()
            return

        # Volume exists
        current_slice = volume[self.slice_idx]
        
        # Check if we need to re-initialize the image
        # 1. No handle
        # 2. Handle exists but axes cleared
        # 3. Shape mismatch (switching views or patients)
        need_reinit = (self.image_handle is None or 
                      len(self.ax.images) == 0 or 
                      self.image_handle.get_array().shape != current_slice.shape)
        
        if need_reinit:
            self.ax.clear() # Clear any text/artifacts
            self.ax.axis('off')
            
            # Re-create image
            self.image_handle = self.ax.imshow(current_slice, cmap='bone', 
                                              vmin=0, vmax=1, interpolation='bilinear')
            
            # Re-create title handle
            self.title_handle = self.ax.set_title('', fontsize=12, fontweight='bold', pad=20)
            
            # Re-create coord text
            self.coord_text = self.ax.text(0.01, 0.99, '', transform=self.ax.transAxes, 
                                          color='yellow', fontsize=10, verticalalignment='top',
                                          bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

        # Ensure slice index is valid
        self.slice_idx = np.clip(self.slice_idx, 0, volume.shape[0] - 1)
        
        # Update image data
        # Note: If we just re-initialized, this sets it again, but that's cheap
        self.image_handle.set_data(volume[self.slice_idx])
        
        # Adjust aspect ratio based on view and metadata
        if self.view == 'coronal':
            # For coronal: pixel_spacing is in-plane (horizontal), slice_thickness is vertical
            pixel_spacing = self.metadata['pixel_spacing'] if isinstance(self.metadata['pixel_spacing'], (int, float)) else self.metadata['pixel_spacing'][0]
            aspect = pixel_spacing / self.metadata['slice_thickness']
        else:
            # For axial: square pixels
            aspect = 'equal'
        self.ax.set_aspect(aspect)
        
        # Update title text
        title = f"{patient_id} - {self.side.upper()} ear - {self.view.upper()} view\n"
        title += f"Slice {self.slice_idx + 1}/{volume.shape[0]}"
        title += f" | Shape: {volume.shape}"
        title += f"\nPixel spacing: {self.metadata['pixel_spacing']} mm"
        title += f" | Slice thickness: {self.metadata['slice_thickness']} mm"
        
        self.title_handle.set_text(title)
        
        # Redraw
        self.fig.canvas.draw_idle()
    
    def on_scroll(self, event):
        """Handle mouse scroll events"""
        volume = self.get_current_volume()
        if volume is None: return
        
        if event.button == 'up':
            self.slice_idx = min(self.slice_idx + 1, volume.shape[0] - 1)
        elif event.button == 'down':
            self.slice_idx = max(self.slice_idx - 1, 0)
        
        self.update_display()
    
    def on_key(self, event):
        """Handle key press events"""
        volume = self.get_current_volume()
        
        if event.key == 'q':
            plt.close()
            return
        elif event.key == 'l':
            self.side = 'left'
            self.load_current_data()
            self.update_display()
            return
        elif event.key == 'r':
            self.side = 'right'
            self.load_current_data()
            self.update_display()
            return
        elif event.key == 'n':
            self.patient_idx = (self.patient_idx + 1) % len(self.patients)
            self.side = 'left'  # Reset to left ear
            self.load_current_data()
            self.update_display()
            return
        elif event.key == 'p':
            self.patient_idx = (self.patient_idx - 1) % len(self.patients)
            self.side = 'left'  # Reset to left ear
            self.load_current_data()
            self.update_display()
            return
            
        if volume is None: return

        if event.key == 'right':
            self.slice_idx = min(self.slice_idx + 1, volume.shape[0] - 1)
        elif event.key == 'left':
            self.slice_idx = max(self.slice_idx - 1, 0)
        elif event.key == 'up':
            self.slice_idx = min(self.slice_idx + 10, volume.shape[0] - 1)
        elif event.key == 'down':
            self.slice_idx = max(self.slice_idx - 10, 0)
        elif event.key == 'pageup':
            self.slice_idx = min(self.slice_idx + 50, volume.shape[0] - 1)
        elif event.key == 'pagedown':
            self.slice_idx = max(self.slice_idx - 50, 0)
        elif event.key == 'home':
            self.slice_idx = 0
        elif event.key == 'end':
            self.slice_idx = volume.shape[0] - 1
        elif event.key == 'a':
            self.view = 'axial'
            self.slice_idx = self.axial.shape[0] // 2
        elif event.key == 'c':
            self.view = 'coronal'
            self.slice_idx = self.coronal.shape[0] // 2
        
        self.update_display()
    
    def show(self):
        """Show the viewer"""
        plt.show()


def main():
    """Main function"""
    print("="*60)
    print("INTERACTIVE VIEWER - Temporal Bone HRCT")
    print("="*60)
    print("\nControls:")
    print("  ← / → : Navigate slices (one at a time)")
    print("  ↑ / ↓ : Navigate slices (10 at a time)")
    print("  PgUp/PgDn : Navigate slices (50 at a time)")
    print("  Home/End : Jump to first/last slice")
    print("  Scroll wheel : Navigate slices")
    print("  a / c : Switch to Axial / Coronal view")
    print("  l / r : Switch to Left / Right ear")
    print("  n / p : Next / Previous patient")
    print("  q     : Quit")
    print("\n" + "="*60)
    
    viewer = InteractiveViewer()
    viewer.show()


if __name__ == '__main__':
    main()
