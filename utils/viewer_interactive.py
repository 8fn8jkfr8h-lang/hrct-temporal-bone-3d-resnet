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
        
        self.setup_figure()
        self.load_current_data()
        
    def load_current_data(self):
        """Load data for current patient and side"""
        patient_id = self.patients[self.patient_idx].name
        side_dir = self.patients[self.patient_idx] / self.side
        
        # Load volumes
        self.axial = np.load(side_dir / 'axial_volume.npy')
        self.coronal = np.load(side_dir / 'coronal_volume.npy')
        
        # Flip coronal view vertically (axis 1) so Head is Up
        # Data is (Y, Z, X), we flip Z
        self.coronal = np.flip(self.coronal, axis=1)
        
        # Load metadata
        with open(side_dir / 'metadata.json', 'r') as f:
            self.metadata = json.load(f)
        
        # Reset slice index to middle
        self.slice_idx = self.get_current_volume().shape[0] // 2
        
        print(f"\nLoaded: {patient_id} - {self.side.upper()} ear")
        print(f"Axial shape: {self.axial.shape}")
        print(f"Coronal shape: {self.coronal.shape}")
        
    def get_current_volume(self):
        """Get currently selected volume"""
        return self.axial if self.view == 'axial' else self.coronal
    
    def setup_figure(self):
        """Setup matplotlib figure with optimized rendering"""
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('scroll_event', self.on_scroll)
        
        # Initial display
        volume = self.get_current_volume()
        self.image_handle = self.ax.imshow(volume[self.slice_idx], cmap='bone', 
                                           vmin=0, vmax=1, interpolation='bilinear')
        self.ax.axis('off')
        
        # Title
        self.title_handle = self.ax.set_title('', fontsize=12, fontweight='bold', pad=20)
        
        # Controls text
        controls = "Controls: ←/→ slices | Scroll wheel | a/c axial/coronal | l/r left/right | n/p next/prev patient | q quit"
        self.fig.text(0.5, 0.02, controls, ha='center', fontsize=10, 
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        self.update_display()
        
    def update_display(self):
        """Update the display (optimized for speed)"""
        # Ensure image and title handles are initialized
        if self.image_handle is None or self.title_handle is None:
            return

        volume = self.get_current_volume()
        
        # Ensure slice index is valid
        self.slice_idx = np.clip(self.slice_idx, 0, volume.shape[0] - 1)
        
        # Update image data (faster than clearing and redrawing)
        self.image_handle.set_data(volume[self.slice_idx])
        
        # Update title
        patient_id = self.patients[self.patient_idx].name
        title = f"{patient_id} - {self.side.upper()} ear - {self.view.upper()} view\n"
        title += f"Slice {self.slice_idx + 1}/{volume.shape[0]}"
        title += f" | Shape: {volume.shape}"
        title += f"\nPixel spacing: {self.metadata['pixel_spacing']} mm"
        title += f" | Slice thickness: {self.metadata['slice_thickness']} mm"
        
        self.title_handle.set_text(title)
        
        # Redraw only the changed parts
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
    
    def on_scroll(self, event):
        """Handle mouse scroll events"""
        if event.button == 'up':
            self.slice_idx = min(self.slice_idx + 1, self.get_current_volume().shape[0] - 1)
        elif event.button == 'down':
            self.slice_idx = max(self.slice_idx - 1, 0)
        
        self.update_display()
    
    def on_key(self, event):
        """Handle key press events"""
        volume = self.get_current_volume()
        
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
        elif event.key == 'l':
            self.side = 'left'
            self.load_current_data()
        elif event.key == 'r':
            self.side = 'right'
            self.load_current_data()
        elif event.key == 'n':
            self.patient_idx = (self.patient_idx + 1) % len(self.patients)
            self.load_current_data()
        elif event.key == 'p':
            self.patient_idx = (self.patient_idx - 1) % len(self.patients)
            self.load_current_data()
        elif event.key == 'q':
            plt.close()
            return
        else:
            return
        
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
