"""
Data Validation Utilities
Comprehensive quality checks on processed volumes
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List


def validate_volume(volume_path: Path, volume_type: str) -> Dict:
    """Validate a single volume file"""
    
    results = {
        'path': str(volume_path),
        'type': volume_type,
        'exists': volume_path.exists(),
        'valid': False,
        'shape': None,
        'dtype': None,
        'value_range': None,
        'has_nan': None,
        'has_inf': None,
        'issues': []
    }
    
    if not results['exists']:
        results['issues'].append('File does not exist')
        return results
    
    try:
        volume = np.load(volume_path)
        
        results['shape'] = volume.shape
        results['dtype'] = str(volume.dtype)
        results['value_range'] = (float(volume.min()), float(volume.max()))
        results['has_nan'] = bool(np.isnan(volume).any())
        results['has_inf'] = bool(np.isinf(volume).any())
        
        # Check for issues
        if results['has_nan']:
            results['issues'].append('Contains NaN values')
        
        if results['has_inf']:
            results['issues'].append('Contains Inf values')
        
        # Check value range (should be [0, 1] after windowing)
        vmin, vmax = results['value_range']
        if vmin < -0.01 or vmax > 1.01:
            results['issues'].append(f'Values outside [0,1]: [{vmin:.3f}, {vmax:.3f}]')
        
        # Check shape
        if volume_type == 'axial':
            if len(volume.shape) != 3:
                results['issues'].append(f'Expected 3D, got {volume.shape}')
            elif volume.shape[1] != 768:
                results['issues'].append(f'Expected height=768, got {volume.shape[1]}')
        
        elif volume_type == 'coronal':
            if len(volume.shape) != 3:
                results['issues'].append(f'Expected 3D, got {volume.shape}')
        
        results['valid'] = len(results['issues']) == 0
        
    except Exception as e:
        results['issues'].append(f'Error loading: {str(e)}')
    
    return results


def validate_metadata(metadata_path: Path) -> Dict:
    """Validate metadata file"""
    
    results = {
        'path': str(metadata_path),
        'exists': metadata_path.exists(),
        'valid': False,
        'required_fields': [],
        'issues': []
    }
    
    if not results['exists']:
        results['issues'].append('File does not exist')
        return results
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        required = [
            'patient_id', 'side', 'num_slices', 'pixel_spacing',
            'slice_thickness', 'axial_shape', 'coronal_shape'
        ]
        
        for field in required:
            if field in metadata:
                results['required_fields'].append(field)
            else:
                results['issues'].append(f'Missing field: {field}')
        
        if 'pixel_spacing' in metadata:
            if len(metadata['pixel_spacing']) != 2:
                results['issues'].append('pixel_spacing should have 2 values')
        
        if 'side' in metadata:
            if metadata['side'] not in ['L', 'R']:
                results['issues'].append(f"Invalid side: {metadata['side']}")
        
        results['valid'] = len(results['issues']) == 0
        
    except Exception as e:
        results['issues'].append(f'Error loading: {str(e)}')
    
    return results


def validate_patient(patient_dir: Path) -> Dict:
    """Validate all data for one patient"""
    
    patient_id = patient_dir.name
    results = {
        'patient_id': patient_id,
        'valid': True,
        'ears': {}
    }
    
    for side in ['left', 'right']:
        side_dir = patient_dir / side
        
        if not side_dir.exists():
            results['valid'] = False
            results['ears'][side] = {'error': 'Directory does not exist'}
            continue
        
        ear_results = {
            'axial': validate_volume(side_dir / 'axial_volume.npy', 'axial'),
            'coronal': validate_volume(side_dir / 'coronal_volume.npy', 'coronal'),
            'metadata': validate_metadata(side_dir / 'metadata.json')
        }
        
        ear_valid = all([
            ear_results['axial']['valid'],
            ear_results['coronal']['valid'],
            ear_results['metadata']['valid']
        ])
        
        ear_results['valid'] = ear_valid
        results['ears'][side] = ear_results
        
        if not ear_valid:
            results['valid'] = False
    
    return results



def run_validation(data_dir='processed_data') -> List[Dict]:
    """
    Run validation on all processed data
    
    Returns:
        List of validation results for each patient
    """
    
    print("="*80)
    print("DATA VALIDATION")
    print("="*80)
    
    data_dir = Path(data_dir)
    
    if not data_dir.exists():
        print(f"\n❌ ERROR: {data_dir} does not exist")
        print("Run pipeline/phase1_dicom_ingestion.py first")
        return []
    
    patients = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    
    print(f"\nValidating {len(patients)} patients...\n")
    
    all_results = []
    valid_count = 0
    
    for patient_dir in patients:
        results = validate_patient(patient_dir)
        all_results.append(results)
        
        if results['valid']:
            valid_count += 1
    
    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nTotal patients: {len(all_results)}")
    print(f"Valid: {valid_count}")
    print(f"Invalid: {len(all_results) - valid_count}")
    
    # Detailed results
    print("\n" + "="*80)
    print("DETAILS")
    print("="*80)
    
    for results in all_results:
        patient_id = results['patient_id']
        status = "✅ PASS" if results['valid'] else "❌ FAIL"
        
        print(f"\n{patient_id}: {status}")
        
        for side, ear_results in results['ears'].items():
            if 'error' in ear_results:
                print(f"  {side.upper()}: ❌ {ear_results['error']}")
                continue
            
            ear_status = "✅" if ear_results['valid'] else "❌"
            print(f"  {side.upper()}: {ear_status}")
            
            # Axial
            axial = ear_results['axial']
            if axial['valid']:
                print(f"    Axial:   ✅ {axial['shape']} {axial['dtype']} [{axial['value_range'][0]:.3f}, {axial['value_range'][1]:.3f}]")
            else:
                print(f"    Axial:   ❌ {', '.join(axial['issues'])}")
            
            # Coronal
            coronal = ear_results['coronal']
            if coronal['valid']:
                print(f"    Coronal: ✅ {coronal['shape']} {coronal['dtype']} [{coronal['value_range'][0]:.3f}, {coronal['value_range'][1]:.3f}]")
            else:
                print(f"    Coronal: ❌ {', '.join(coronal['issues'])}")
            
            # Metadata
            metadata = ear_results['metadata']
            if metadata['valid']:
                print(f"    Metadata: ✅")
            else:
                print(f"    Metadata: ❌ {', '.join(metadata['issues'])}")
    
    # Overall status
    print("\n" + "="*80)
    if valid_count == len(all_results):
        print("✅ ALL VALIDATION CHECKS PASSED")
    else:
        print(f"⚠️  {len(all_results) - valid_count} PATIENT(S) FAILED")
    print("="*80 + "\n")
    
    return all_results


if __name__ == '__main__':
    run_validation()
