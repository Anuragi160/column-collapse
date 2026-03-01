import subprocess
import os

OUTPUT_DIR = '/home/vision/Desktop/DEM-ML-250'
YADE = 'yade'

# Delete old npz files
for f in ['train.npz', 'valid.npz', 'test.npz']:
    path = os.path.join(OUTPUT_DIR, f)
    if os.path.exists(path):
        os.remove(path)
        print(f"Deleted {path}")

splits = []
for seed in range(1, 21):
    splits.append((seed, 'train'))
for seed in range(21, 24):
    splits.append((seed, 'valid'))
for seed in range(24, 27):
    splits.append((seed, 'test'))

total = len(splits)
for i, (seed, split) in enumerate(splits):
    print(f"\n[{i+1}/{total}] Running SEED={seed} SPLIT={split}")
    cmd = [YADE, '-x', 'dem_simulation.py', str(seed), split, OUTPUT_DIR]
    subprocess.run(cmd, cwd=OUTPUT_DIR)
    print(f"[{i+1}/{total}] Done SEED={seed} SPLIT={split}")

print("\n" + "="*50)
print("FINAL VERIFICATION")
print("="*50)
import numpy as np
for split in ['train', 'valid', 'test']:
    d = np.load(os.path.join(OUTPUT_DIR, f'{split}.npz'), allow_pickle=True)
    arr = d['gns_data']
    print(f"{split}: {len(arr)} trajectories, shape={arr[0][0].shape}")
