"""
Generate Processing Report
Detailed statistics and summary of processed data
"""

import json
import pandas as pd
from pathlib import Path


def generate_report(data_dir='processed_data', labels_file='labels.csv', output_file='processing_report.csv'):
    """Generate comprehensive processing report"""
    
    print("="*80)
    print("PROCESSING REPORT")
    print("="*80)
    
    # Load labels
    labels_df = pd.read_csv(labels_file)
    
    # Get processed patients
    data_dir = Path(data_dir)
    patients = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    
    print(f"\nPatients processed: {len(patients)}")
    print(f"Total ears: {len(patients) * 2}")
    
    # Collect data
    all_data = []
    
    for patient_dir in patients:
        patient_id = patient_dir.name
        
        for side in ['left', 'right']:
            side_dir = patient_dir / side
            
            if not side_dir.exists():
                continue
            
            # Load metadata
            with open(side_dir / 'metadata.json', 'r') as f:
                metadata = json.load(f)
            
            # Get labels
            side_letter = 'L' if side == 'left' else 'R'
            label_row = labels_df[
                (labels_df['patient_id'] == patient_id) & 
                (labels_df['ear'] == side_letter)
            ]
            
            if not label_row.empty:
                label_row = label_row.iloc[0]
                cholesteatoma = label_row['cholesteatoma']
                ossicular = label_row['ossicular_discontinuity']
                facial = label_row['facial_dehiscence']
                surgery = label_row['surgery_type']
                status = label_row['exclusion_status']
            else:
                cholesteatoma = ossicular = facial = surgery = status = 'N/A'
            
            all_data.append({
                'patient_id': patient_id,
                'ear': side_letter,
                'axial_slices': metadata['axial_shape'][0],
                'axial_height': metadata['axial_shape'][1],
                'axial_width': metadata['axial_shape'][2],
                'coronal_slices': metadata['coronal_shape'][0],
                'coronal_height': metadata['coronal_shape'][1],
                'coronal_width': metadata['coronal_shape'][2],
                'pixel_spacing_x': metadata['pixel_spacing'][0],
                'pixel_spacing_y': metadata['pixel_spacing'][1],
                'slice_thickness': metadata['slice_thickness'],
                'original_slices': metadata['num_slices'],
                'cholesteatoma': cholesteatoma,
                'ossicular': ossicular,
                'facial': facial,
                'surgery': surgery,
                'status': status
            })
    
    df = pd.DataFrame(all_data)
    
    # Statistics
    print("\n" + "="*80)
    print("STATISTICS")
    print("="*80)
    
    print(f"\nTotal ears: {len(df)}")
    
    included = df[df['status'] == 'include']
    print(f"Included: {len(included)}")
    print(f"Excluded: {len(df) - len(included)}")
    
    if len(included) > 0:
        print("\n--- Pathology Distribution (Included Only) ---")
        
        # Cholesteatoma
        chol_valid = included[included['cholesteatoma'].notna() & (included['cholesteatoma'] != '')]
        if len(chol_valid) > 0:
            chol_pos = (chol_valid['cholesteatoma'] == 1).sum()
            chol_neg = (chol_valid['cholesteatoma'] == 0).sum()
            print(f"\nCholesteatoma:")
            print(f"  Positive: {chol_pos}/{len(chol_valid)} ({chol_pos/len(chol_valid)*100:.1f}%)")
            print(f"  Negative: {chol_neg}/{len(chol_valid)} ({chol_neg/len(chol_valid)*100:.1f}%)")
        
        # Ossicular
        ossi_valid = included[included['ossicular'].notna() & (included['ossicular'] != '')]
        if len(ossi_valid) > 0:
            ossi_pos = (ossi_valid['ossicular'] == 1).sum()
            ossi_neg = (ossi_valid['ossicular'] == 0).sum()
            print(f"\nOssicular Discontinuity:")
            print(f"  Positive: {ossi_pos}/{len(ossi_valid)} ({ossi_pos/len(ossi_valid)*100:.1f}%)")
            print(f"  Negative: {ossi_neg}/{len(ossi_valid)} ({ossi_neg/len(ossi_valid)*100:.1f}%)")
        
        # Facial
        facial_valid = included[included['facial'].notna() & (included['facial'] != '')]
        if len(facial_valid) > 0:
            facial_pos = (facial_valid['facial'] == 1).sum()
            facial_neg = (facial_valid['facial'] == 0).sum()
            print(f"\nFacial Nerve Dehiscence:")
            print(f"  Positive: {facial_pos}/{len(facial_valid)} ({facial_pos/len(facial_valid)*100:.1f}%)")
            print(f"  Negative: {facial_neg}/{len(facial_valid)} ({facial_neg/len(facial_valid)*100:.1f}%)")
    
    # Volume statistics
    print("\n--- Volume Dimensions ---")
    print(f"Axial slices: {df['axial_slices'].min()}-{df['axial_slices'].max()} (mean: {df['axial_slices'].mean():.1f})")
    print(f"Coronal slices: {df['coronal_slices'].min()}-{df['coronal_slices'].max()} (mean: {df['coronal_slices'].mean():.1f})")
    
    print("\n--- Pixel Spacing ---")
    print(f"X: {df['pixel_spacing_x'].min():.4f}-{df['pixel_spacing_x'].max():.4f} mm")
    print(f"Y: {df['pixel_spacing_y'].min():.4f}-{df['pixel_spacing_y'].max():.4f} mm")
    print(f"Z: {df['slice_thickness'].min():.2f}-{df['slice_thickness'].max():.2f} mm")
    
    # Storage
    print("\n--- Storage ---")
    total_size = 0
    for patient_dir in patients:
        for side in ['left', 'right']:
            side_dir = patient_dir / side
            if side_dir.exists():
                axial_size = (side_dir / 'axial_volume.npy').stat().st_size
                coronal_size = (side_dir / 'coronal_volume.npy').stat().st_size
                total_size += axial_size + coronal_size
    
    print(f"Total: {total_size / (1024**3):.2f} GB")
    print(f"Per ear: {total_size / len(df) / (1024**2):.1f} MB")
    
    # Save
    df.to_csv(output_file, index=False)
    print(f"\nReport saved: {output_file}")
    
    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    generate_report()
