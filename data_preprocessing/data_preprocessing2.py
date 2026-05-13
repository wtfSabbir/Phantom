"""
data_preprocessing2.py - Convert LAZ tiles to Pointcept npy format for v2 training.

Reads GT labels from gt folder and v1_pred labels from v1_pred folder.
Both folders must have matching filenames.

Output per tile:
    coord.npy       (N, 3) float32  - xyz centered by tile mean
    color.npy       (N, 1) float32  - intensity normalized [0, 1]
    segment.npy     (N,)   int64    - ground truth labels (remapped)
    v1_pred.npy     (N,)   int64    - v1 predicted labels (remapped)

Usage:
    python data_preprocessing2.py --config config_v2.yaml
"""

import os
import argparse
import numpy as np
import laspy
from pathlib import Path
from tqdm import tqdm
import yaml

CLASS_MAP = {
    0: 0,  # Unclassified -> ignore
    1: 1,    # Ground
    2: 2,    # Pole
    3: 3,    # Sign
    4: 4,    # Bollard
    5: 5,    # Trunk
    6: 6,    # Vegetation
    7: 7,    # Building
    8: 255,  # RIEN -> ignore
    9: 8,    # Fence
    10: 9,   # Gate
}


def remap_labels(labels, class_map, ignore_index=255):
    remapped = np.full_like(labels, ignore_index, dtype=np.int64)
    for orig_id, new_id in class_map.items():
        remapped[labels == orig_id] = new_id
    return remapped


def process_file(gt_path, v1_path, output_dir, split, class_map, ignore_index=255):
    stem = Path(gt_path).stem

    try:
        gt_las = laspy.read(gt_path)
    except Exception as e:
        print(f"  [ERROR] Cannot read GT {gt_path}: {e}")
        return

    try:
        v1_las = laspy.read(v1_path)
    except Exception as e:
        print(f"  [ERROR] Cannot read V1_Pred {v1_path}: {e}")
        return

    tile_dir = os.path.join(output_dir, split, stem)
    os.makedirs(tile_dir, exist_ok=True)

    # XYZ from GT file
    x = np.array(gt_las.x, dtype=np.float32)
    y = np.array(gt_las.y, dtype=np.float32)
    z = np.array(gt_las.z, dtype=np.float32)

    coord = np.stack([x, y, z], axis=1)
    coord -= coord.mean(axis=0)

    # Intensity from GT file
    if hasattr(gt_las, "intensity"):
        intensity = np.array(gt_las.intensity, dtype=np.float32) / 65535.0
    else:
        intensity = np.zeros(len(x), dtype=np.float32)
    color = intensity.reshape(-1, 1)

    # GT labels from GT file classification field
    gt_labels = np.array(gt_las.classification, dtype=np.int64)
    segment   = remap_labels(gt_labels, class_map, ignore_index)

    # V1 labels from V1_Pred file classification field
    v1_labels = np.array(v1_las.classification, dtype=np.int64)
    v1_pred   = remap_labels(v1_labels, class_map, ignore_index)

    np.save(os.path.join(tile_dir, "coord.npy"),   coord.astype(np.float32))
    np.save(os.path.join(tile_dir, "color.npy"),   color.astype(np.float32))
    np.save(os.path.join(tile_dir, "segment.npy"), segment.astype(np.int64))
    np.save(os.path.join(tile_dir, "v1_pred.npy"), v1_pred.astype(np.int64))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config_v2.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train_gt_dir  = cfg["train_gt_dir"]
    val_gt_dir    = cfg["val_gt_dir"]
    train_v1_dir  = cfg["train_v1_dir"]
    val_v1_dir    = cfg["val_v1_dir"]
    output_dir    = cfg["output_npy_dir"]
    ignore_idx    = cfg.get("ignore_index", 255)

    for split, gt_dir, v1_dir in [
        ("train", train_gt_dir, train_v1_dir),
        ("val",   val_gt_dir,   val_v1_dir),
    ]:
        gt_files = sorted(Path(gt_dir).glob("*.la[sz]"))
        print(f"\n[{split.upper()}] Processing {len(gt_files)} files")

        for gt_path in tqdm(gt_files, desc=split):
            v1_path = Path(v1_dir) / gt_path.name
            if not v1_path.exists():
                print(f"  [WARN] No matching V1_Pred file for {gt_path.name}, skipping")
                continue
            process_file(str(gt_path), str(v1_path), output_dir,
                         split, CLASS_MAP, ignore_idx)

    print("\nDone. Each tile now has coord, color, segment, v1_pred npy files.")


if __name__ == "__main__":
    main()