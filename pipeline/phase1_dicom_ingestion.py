"""
Phase 1: DICOM Ingestion and Preprocessing
Temporal Bone HRCT Project

This script:
1. Loads DICOM files with proper slice sorting
2. Extracts metadata (spacing, thickness)
3. Converts to Hounsfield Units
4. Applies bone windowing
5. Reconstructs coronal views
6. Splits left/right temporal bones
7. Saves processed volumes and metadata
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from utils.dicom_processor import DICOMProcessor


def main():
    """Main execution function"""
    
    # Configuration
    INPUT_DIR = 'dicom_input'
    OUTPUT_DIR = 'processed_data'
    LABELS_FILE = 'labels.csv'
    
    print("="*60)
    print("PHASE 1: DICOM INGESTION AND PREPROCESSING")
    print("="*60)
    
    # Initialize processor
    processor = DICOMProcessor(INPUT_DIR, OUTPUT_DIR, LABELS_FILE)
    
    # Process all patients
    results = processor.process_all_patients()
    
    print(f"\n{'='*60}")
    print("Phase 1 complete!")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
