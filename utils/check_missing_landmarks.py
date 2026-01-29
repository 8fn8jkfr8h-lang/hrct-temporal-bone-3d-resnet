"""Check which patients have missing landmarks"""
import json
from pathlib import Path

landmarks_dir = Path('landmarks_detected')

print("Identifying patients with missing critical landmarks:")
print("=" * 60)

problems = []

for lf in sorted(landmarks_dir.glob('*/*/landmark_coords.json')):
    patient_id = lf.parent.parent.name
    side = lf.parent.name
    
    with open(lf) as f:
        data = json.load(f)
    
    landmarks = data.get('landmarks', {})
    apex = landmarks.get('apex')
    basal = landmarks.get('basal')
    rw = landmarks.get('round_window')
    
    missing = []
    if apex is None:
        missing.append('apex')
    if basal is None:
        missing.append('basal')
    if rw is None:
        missing.append('round_window')
        
    if missing:
        problems.append(f"{patient_id}/{side}: Missing {missing}")
        
if problems == []:
    print("No missing landmarks found.")
    
else:
    for p in problems:
        print(p)
