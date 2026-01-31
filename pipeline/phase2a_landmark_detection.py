"""
Phase 2A: Automatic Landmark Detection (OPTIMIZED)

Steps:
1. Load Phase 1 processed volume (numpy).
2. Wrap it in a SMICNet-compatible `Volume` object (requires neuclid/pyrr).
3. Load SMICNet model.
4. Perform Coarse Grid Search (to find approx cochlea location).
5. Perform Fine Sliding Window Search (around the coarse match).
6. Extract centroids for Apex, Basal, and Round Window.

Optimizations:
- Mixed precision (FP16) for Tensor Core acceleration
- Batched coarse search (major speedup)
- Increased batch size for RTX 3050
- Direct model call instead of predict() for lower latency
"""

import sys
from pathlib import Path
import numpy as np
import json
import time
import gc
import tensorflow as tf
from scipy import ndimage as ndi

# --- GPU Configuration with Mixed Precision ---
print("=" * 60)
print("GPU CONFIGURATION")
print("=" * 60)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"✅ GPU DETECTED: {len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        print(f"   Device: {gpus[0].name}")
        
        # Enable Mixed Precision for Tensor Core acceleration (RTX 3050 has Tensor Cores)
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print("✅ Mixed Precision (FP16) ENABLED - Tensor Core acceleration active")
    except RuntimeError as e:
        print(f"❌ GPU Error: {e}")
else:
    print("⚠️  NO GPU DETECTED. Running on CPU (this will be slow).")

# Add smicnet to path
SMICNET_ROOT = Path('./smicnet').absolute()
if str(SMICNET_ROOT) not in sys.path:
    sys.path.insert(0, str(SMICNET_ROOT))

# SMICNet imports
try:
    from SMICNet import SMICNet_build # type: ignore
except ImportError as e:
    print(f"Error importing SMICNet modules: {e}")
    sys.exit(1)

# --- Helper Functions ---

def sliding_range_creating(range_set=5):
    """Create local search grid offset"""
    x, y, z = (
        np.arange(-range_set, range_set),
        np.arange(-range_set, range_set),
        np.arange(-range_set, range_set),
    )
    xx, yy, zz = np.meshgrid(x, y, z)
    return np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))


class SMICNetInference:
    """Optimized SMICNet inference with batched processing"""
    
    # Optimized batch sizes for RTX 3050 6GB
    COARSE_BATCH_SIZE = 512
    FINE_BATCH_SIZE = 1024
    
    def __init__(self, weights_path):
        self.model = SMICNet_build()
        self.model.load_weights(str(weights_path))
        print(f"✅ Loaded SMICNet weights from {weights_path}")
        
        # Warmup the model with a dummy batch to initialize CUDA kernels
        print("   Warming up model...")
        dummy_input = np.zeros((1, 81, 81, 1), dtype=np.float32)
        _ = self.model(dummy_input, training=False)
        print("   Model ready.")
        
    def _extract_batch_slices(self, volume_data, coords):
        """Extract multiple 81x81 slices at once for batched processing"""
        batch = []
        valid_coords = []
        
        for z, y, x in coords:
            z_start, z_end = z - 40, z + 41
            y_start, y_end = y - 40, y + 41
            
            # Check bounds
            if z_start < 0 or z_end > volume_data.shape[0]: continue
            if y_start < 0 or y_end > volume_data.shape[1]: continue
            if x < 0 or x >= volume_data.shape[2]: continue
            
            # Sagittal slice extraction
            # Current Volume Shape: (Z, Y, X)
            # Extracted Slice: volume[Z, Y, fixed_X] -> Shape (Z, Y) -> (Inf-Sup, Ant-Post)
            slice_img = volume_data[z_start:z_end, y_start:y_end, x]
            
            if slice_img.shape == (81, 81):
                # CRITICAL FIX: Transpose to match SMICNet training orientation
                # Training expects: (Y, Z) -> (Ant-Post, Inf-Sup)
                slice_transposed = slice_img.T 
                
                batch.append(slice_transposed[:, :, np.newaxis])
                valid_coords.append((z, y, x))
        
        if not batch:
            return None, []
        
        return np.array(batch, dtype=np.float32), valid_coords
        
    def predict_volume(self, volume_data, side='left', coarse_step=5):
        """
        Full pipeline to detect landmarks in a volume.
        
        OPTIMIZED: Uses batched inference for both coarse and fine search.
        """
        total_start_time = time.time()
        
        # 0. Handle Right Ear Mirroring
        # SMICNet expects Left ear orientation. If Right, flip X axis.
        original_volume = volume_data
        if side == 'right':
            print("   ↔️  Right ear detected: Flipping volume for inference...")
            volume_data = np.flip(volume_data, axis=2)
            
        width = volume_data.shape[2]
        
        # 1. Automatic Initialization (Coarse Search) - BATCHED
        print("\n📍 Step 1: Finding Region of Interest (Batched Coarse Search)...")
        coarse_start = time.time()
        
        center_z, center_y, center_x = np.array(volume_data.shape) // 2
        
        # Search the ENTIRE valid volume (margins of 45 needed for 81x81 patch)
        z_range = list(range(45, volume_data.shape[0]-45, coarse_step))
        y_range = list(range(45, volume_data.shape[1]-45, coarse_step))
        x_range = list(range(45, volume_data.shape[2]-45, coarse_step))
        
        # Generate all coordinates to search
        all_coords = [(z, y, x) for x in x_range for y in y_range for z in z_range]
        print(f"   Searching {len(all_coords)} locations with batch size {self.COARSE_BATCH_SIZE}...")
        
        # Store Top-K Candidates to avoid getting stuck in a local maximum
        K = 5
        top_candidates = [] # list of (prob, (z,y,x))
        
        # Process in batches
        for batch_start in range(0, len(all_coords), self.COARSE_BATCH_SIZE):
            batch_coords = all_coords[batch_start:batch_start + self.COARSE_BATCH_SIZE]
            batch_arr, valid_coords = self._extract_batch_slices(volume_data, batch_coords)
            
            if batch_arr is None:
                continue
            
            # Direct model call
            preds = self.model(batch_arr, training=False).numpy()
            
            for i, (z, y, x) in enumerate(valid_coords):
                # Sum of probabilities of landmarks (Apex + Basal + RW)
                landmark_prob = float(preds[i, 0] + preds[i, 1] + preds[i, 2])
                
                if len(top_candidates) < K:
                    top_candidates.append((landmark_prob, (z, y, x)))
                    top_candidates.sort(reverse=True, key=lambda x: x[0])
                elif landmark_prob > top_candidates[-1][0]:
                    top_candidates[-1] = (landmark_prob, (z, y, x))
                    top_candidates.sort(reverse=True, key=lambda x: x[0])
        
        coarse_time = time.time() - coarse_start
        
        # 2. Fine Search (Sliding Window around Best Seeds) - BATCHED
        print("\n📍 Step 2: Fine Sliding Window Search (Multi-Seed)...")
        fine_start = time.time()
        
        best_overall_landmarks = None
        best_overall_confidence = 0.0
        
        # Try top candidates until we get a good detection
        search_candidates = top_candidates if top_candidates else [(0, (center_z, center_y, center_x))]
        
        for rank, (prob, seed) in enumerate(search_candidates):
            print(f"   🔎 Checking Candidate {rank+1}/{len(search_candidates)}: Seed {seed} (Coarse Prob: {prob:.2f})")
            
            # Run fine search
            landmarks, confidence = self._fine_search(volume_data, seed)
            
            # Check if this result is better
            if confidence > best_overall_confidence:
                best_overall_confidence = confidence
                best_overall_landmarks = landmarks
                print(f"      ✅ New best result! (Conf: {confidence:.2f})")
                
                # If very confident, stop early
                if confidence > 2.5: # Max possible is 3.0
                    print("      ✨ High confidence result found. Stopping search.")
                    break
            else:
                print(f"      ❌ Result inferior (Conf: {confidence:.2f})")

        fine_time = time.time() - fine_start
        
        # Use best found
        landmarks = best_overall_landmarks if best_overall_landmarks else {k:None for k in ['apex','basal','round_window']}
        
        # 3. Restore Coordinates (Un-flip)
        if side == 'right':
            print("   ↔️  Restoring Right ear coordinates...")
            for k, v in landmarks.items():
                if v is not None:
                    # v is (z, y, x)
                    # Flip x: new_x = width - 1 - old_x
                    landmarks[k] = [v[0], v[1], width - 1 - v[2]]
        
        total_time = time.time() - total_start_time
        print(f"\n⏱️  Total detection time: {total_time:.2f}s")
        
        return landmarks, {
            'coarse_search_time': coarse_time,
            'fine_search_time': fine_time,
            'total_time': total_time
        }

    def _fine_search(self, volume_data, seed):
        """
        Perform dense search around seed.
        Returns: (landmarks_dict, total_confidence_score)
        """
        z_seed, y_seed, x_seed = seed
        range_set = 20
        local_size = 2 * range_set
        
        # Probability maps
        prob_map_apex = np.zeros((local_size, local_size, local_size), dtype=np.float32)
        prob_map_basal = np.zeros((local_size, local_size, local_size), dtype=np.float32)
        prob_map_rw = np.zeros((local_size, local_size, local_size), dtype=np.float32)
        
        offsets = sliding_range_creating(range_set)
        
        # Generate all coordinates
        all_coords = [(z_seed + dz, y_seed + dy, x_seed + dx) for dz, dy, dx in offsets]
        
        valid_all_coords = []
        for z, y, x in all_coords:
            if (40 <= z < volume_data.shape[0]-40 and 
                40 <= y < volume_data.shape[1]-40 and 
                40 <= x < volume_data.shape[2]-40):
                valid_all_coords.append((z, y, x))
        
        # Process in batches
        for batch_start in range(0, len(valid_all_coords), self.FINE_BATCH_SIZE):
            batch_coords = valid_all_coords[batch_start:batch_start + self.FINE_BATCH_SIZE]
            batch_arr, valid_coords = self._extract_batch_slices(volume_data, batch_coords)
            
            if batch_arr is None: continue
            
            preds = self.model(batch_arr, training=False).numpy()
            
            for i, (pz, py, px) in enumerate(valid_coords):
                lz = pz - z_seed + range_set
                ly = py - y_seed + range_set
                lx = px - x_seed + range_set
                
                if 0 <= lz < local_size and 0 <= ly < local_size and 0 <= lx < local_size:
                    prob_map_apex[lz, ly, lx] = preds[i, 0]
                    prob_map_basal[lz, ly, lx] = preds[i, 1]
                    prob_map_rw[lz, ly, lx] = preds[i, 2]
            
        # Extract Centroids and Score
        landmarks = {}
        total_max_prob = 0.0
        
        # Confidence Explanation:
        # The model outputs probabilities for [Apex, Basal, RoundWindow, Background].
        # We sum the *maximum* probability found for each landmark in the search window.
        # Max confidence = 1.0 (Apex) + 1.0 (Basal) + 1.0 (RW) = 3.0.
        for name, pmap in [('apex', prob_map_apex), ('basal', prob_map_basal), ('round_window', prob_map_rw)]:
            threshold = 0.3
            max_val = pmap.max()
            total_max_prob += max_val
            
            pmap[pmap < threshold] = 0
            
            if pmap.max() == 0:
                landmarks[name] = None
            else:
                loc_local = ndi.center_of_mass(pmap)
                loc_global = [l + s - range_set for l, s in zip(loc_local, seed)]
                landmarks[name] = loc_global
            
        return landmarks, total_max_prob


def run_detection():
    """Run landmark detection on all patients with timing"""
    
    print("\n" + "=" * 60)
    print("PHASE 2A: AUTOMATIC LANDMARK DETECTION (OPTIMIZED)")
    print("=" * 60)
    
    phase_start_time = time.time()
    
    # Setup
    weights = SMICNET_ROOT / 'TrainedNetworkWeights' / 'model_epoch_100_11volumes_authorprovided.h5'
    if not weights.exists():
        print(f"❌ Model weights not found at {weights}")
        return

    detector = SMICNetInference(weights)
    
    processed_dir = Path('processed_data')
    output_base = Path('landmarks_detected')
    
    # Timing statistics
    timing_stats = []
    
    # Iterate patients
    patient_count = 0
    for patient_dir in sorted(processed_dir.glob('pt_*')):
        for side in ['left', 'right']:
            vol_path = patient_dir / side / 'axial_volume.npy'
            if not vol_path.exists(): continue
            
            print(f"\n{'─'*60}")
            print(f"Processing {patient_dir.name} - {side}")
            print(f"{'─'*60}")
            
            patient_count += 1
            
            vol = None
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Load volume (inside try block to catch OOM)
                    if vol is None:
                        load_start = time.time()
                        vol = np.load(vol_path)
                        load_time = time.time() - load_start
                        print(f"   📦 Loaded volume {vol.shape} in {load_time:.2f}s")
                    
                    # Update: Pass side for flipping logic
                    landmarks, timing = detector.predict_volume(vol, side=side)
                    
                    # Save
                    out_dir = output_base / patient_dir.name / side
                    out_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Validation check
                    status = 'success' if all(landmarks.values()) else 'partial'
                    if all(v is None for v in landmarks.values()):
                        status = 'failed'
                    
                    result = {
                        'patient_id': patient_dir.name,
                        'side': side,
                        'landmarks': landmarks,
                        'status': status,
                        'timing': timing
                    }
                    
                    with open(out_dir / 'landmark_coords.json', 'w') as f:
                        json.dump(result, f, indent=2)
                    
                    print(f"   💾 Saved to {out_dir / 'landmark_coords.json'}")
                    
                    # Collect timing stats
                    timing_stats.append({
                        'patient': patient_dir.name,
                        'side': side,
                        'load_time': load_time,
                        **timing
                    })
                    
                    # Success, break retry loop
                    break

                except (np.core._exceptions._ArrayMemoryError, MemoryError) as e:
                    print(f"⚠️  Memory Error on attempt {attempt+1}/{max_retries}: {e}")
                    print("   ♻️  Running Garbage Collection...")
                    if vol is not None:
                        del vol
                        vol = None
                    gc.collect()
                    time.sleep(2)  # Give OS time to reclaim
                    if attempt == max_retries - 1:
                        print(f"❌ Failed to process {patient_dir.name} {side} after {max_retries} attempts.")
                except Exception as e:
                    print(f"❌ Error processing {patient_dir.name} {side}: {e}")
                    import traceback
                    traceback.print_exc()
                    break # Don't retry on non-memory errors (logic bugs)
            
            # Cleanup memory after processing side
            if vol is not None:
                del vol
            gc.collect()
    
    # Print timing summary
    phase_total_time = time.time() - phase_start_time
    
    print("\n" + "=" * 60)
    print("PHASE 2A TIMING SUMMARY")
    print("=" * 60)
    
    if timing_stats:
        avg_coarse = np.mean([t['coarse_search_time'] for t in timing_stats])
        avg_fine = np.mean([t['fine_search_time'] for t in timing_stats])
        avg_total = np.mean([t['total_time'] for t in timing_stats])
        
        print(f"\n📊 Per-volume statistics (n={len(timing_stats)}):")
        print(f"   • Avg coarse search: {avg_coarse:.2f}s")
        print(f"   • Avg fine search:   {avg_fine:.2f}s")  
        print(f"   • Avg total:         {avg_total:.2f}s")
    
    print(f"\n⏱️  Phase 2A total time: {phase_total_time:.2f}s ({phase_total_time/60:.1f} min)")
    print(f"   Processed {patient_count} volumes")
    if patient_count > 0:
        print(f"   Average per volume: {phase_total_time/patient_count:.2f}s")


if __name__ == '__main__':
    run_detection()