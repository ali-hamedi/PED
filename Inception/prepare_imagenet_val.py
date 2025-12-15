import os
import shutil
from pathlib import Path

IMAGENET_ROOT = Path("./")  
VAL_DIR = IMAGENET_ROOT / "val"
DEVKIT_DIR = IMAGENET_ROOT / "devkit"

gt_path = DEVKIT_DIR  / "data" / "ILSVRC2012_validation_ground_truth.txt"
with open(gt_path, "r") as f:
    gt = [int(line.strip()) for line in f]

import scipy.io

meta_path = DEVKIT_DIR/ "data" / "meta.mat"
meta = scipy.io.loadmat(meta_path, squeeze_me=True)["synsets"]

leaf = [s for s in meta if int(s[4]) == 0]
leaf = sorted(leaf, key=lambda s: int(s[0]))
wnids = [str(s[1]) for s in leaf]  

assert len(wnids) == 1000, f"Expected 1000 wnids, got {len(wnids)}"
assert len(gt) == 50000, f"Expected 50000 val labels, got {len(gt)}"

for i, label in enumerate(gt, start=1):
    wnid = wnids[label - 1]  
    src = VAL_DIR / f"ILSVRC2012_val_{i:08d}.JPEG"
    dst_dir = VAL_DIR / wnid
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if src.exists() and not dst.exists():
        shutil.move(str(src), str(dst))

print("Done. val/ is now organized into 1000 class folders.")
