"""
postprocess_laz.py - Post-processing for predicted LAZ files.

Two filters applied in sequence:
  1. Majority Vote Filter  — removes isolated noise dots inside large surfaces
                             (e.g. red dots inside green vegetation)
  2. Small Component Filter — removes tiny floating clusters that are clearly wrong
                             (e.g. 5-point "Gate" prediction inside a building wall)

Usage:
    python postprocess_laz.py \
        --input_dir  /mnt/d/PointCloudsFiles/Predictions \
        --output_dir /mnt/d/PointCloudsFiles/Predictions_PostProcessed

Optional tuning:
    --mv_radius      0.15   # Majority vote search radius in metres
    --mv_min_neighbors 5    # Min neighbors needed to trigger reassignment
    --sc_min_points  20     # Min points for a component to be kept
    --sc_radius      0.20   # Radius to consider two points as connected
    --skip_mv               # Skip majority vote filter
    --skip_sc               # Skip small component filter
"""

import os
import argparse
import numpy as np
import laspy
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict
from scipy.spatial import cKDTree


# ==============================================================================
# CLASS DEFINITIONS
# ==============================================================================
CLASS_NAMES = {
    0: "Unclassified",
    1: "Ground",
    2: "Pole",
    3: "Sign",
    4: "Bollard",
    5: "Trunk",
    6: "Vegetation",
    7: "Building",
    9: "Fence",
    10: "Gate",
}

# Classes that are large/dominant — protect these from aggressive filtering
LARGE_CLASSES = {1, 6, 7}  # Ground, Vegetation, Building

# Classes that are small/rare — apply component filter to these
RARE_CLASSES = {2, 3, 4, 5, 9, 10}  # Pole, Sign, Bollard, Trunk, Fence, Gate


# ==============================================================================
# FILTER 1: MAJORITY VOTE
# ==============================================================================
def majority_vote_filter(coord, labels, radius=0.15, min_neighbors=5):
    """
    For each point, look at all neighbors within `radius`.
    If the point's label disagrees with the majority of its neighbors
    AND the majority label is a LARGE class → reassign.

    This specifically targets: isolated wrong-label dots inside large surfaces.
    It will NOT reassign a correct Gate point surrounded by other Gate points.

    Args:
        coord:         (N, 3) float32 coordinates
        labels:        (N,)   int     predicted class IDs
        radius:        float  search radius in metres
        min_neighbors: int    minimum neighbors to trigger reassignment

    Returns:
        filtered labels (N,)
    """
    print(f"  [MV] Building KDTree for {len(coord):,} points...")
    tree = cKDTree(coord)
    filtered = labels.copy()

    batch_size = 50000
    n = len(coord)
    changed = 0

    print(f"  [MV] Running majority vote (radius={radius}m, min_neighbors={min_neighbors})...")
    for start in tqdm(range(0, n, batch_size), desc="  MV batches"):
        end = min(start + batch_size, n)
        batch_coords = coord[start:end]

        results = tree.query_ball_point(batch_coords, r=radius)

        for local_i, neighbors in enumerate(results):
            if len(neighbors) < min_neighbors:
                continue

            orig_i = start + local_i
            current_label = labels[orig_i]

            # Only reassign if current label is NOT a large class
            # We never want to change a Vegetation/Building/Ground point
            # based on a noisy neighbor
            if current_label in LARGE_CLASSES:
                continue

            neighbor_labels = labels[neighbors]
            unique, counts = np.unique(neighbor_labels, return_counts=True)
            majority_label = unique[np.argmax(counts)]

            # Only reassign if majority is a large class AND
            # the current label is clearly outnumbered
            majority_count = counts[np.argmax(counts)]
            if (majority_label in LARGE_CLASSES and
                    majority_label != current_label and
                    majority_count / len(neighbors) > 0.7):  # 70% threshold
                filtered[orig_i] = majority_label
                changed += 1

    print(f"  [MV] Reassigned {changed:,} points ({changed/n*100:.2f}%)")
    return filtered


# ==============================================================================
# FILTER 2: SMALL COMPONENT REMOVAL
# ==============================================================================
def find_components(cls_coords, radius):
    """
    Simple union-find connected components.
    Two points are connected if within `radius` of each other.
    """
    n = len(cls_coords)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    # Find all pairs within radius
    tree = cKDTree(cls_coords)
    pairs = tree.query_pairs(r=radius)
    for a, b in pairs:
        union(a, b)

    # Group indices by component root
    components = defaultdict(list)
    for i in range(n):
        components[find(i)].append(i)

    return components


def remove_small_components(coord, labels, min_points=20, radius=0.20):
    """
    For each rare class, find connected components.
    Remove components with fewer than `min_points` points.
    Reassign removed points to their nearest neighbor's label
    (excluding the same class).

    This removes:
    - 5-point "Gate" blob sitting inside a Building wall
    - Tiny isolated "Pole" predictions in the middle of Ground
    - Small "Sign" fragments that are just noise

    Args:
        coord:      (N, 3) float32 coordinates
        labels:     (N,)   int     predicted class IDs
        min_points: int    components smaller than this are removed
        radius:     float  two points are "connected" if within this distance

    Returns:
        filtered labels (N,)
    """
    filtered = labels.copy()
    global_tree = cKDTree(coord)
    total_removed = 0

    for cls in RARE_CLASSES:
        cls_mask = np.where(labels == cls)[0]
        if len(cls_mask) == 0:
            continue

        cls_coords = coord[cls_mask]
        print(f"  [SC] Class {cls} ({CLASS_NAMES.get(cls, cls)}): "
              f"{len(cls_mask):,} points → finding components...")

        components = find_components(cls_coords, radius)

        small_count = sum(1 for c in components.values() if len(c) < min_points)
        print(f"  [SC]   Found {len(components)} components, "
              f"{small_count} smaller than {min_points} pts")

        for comp_indices in components.values():
            if len(comp_indices) >= min_points:
                continue  # Keep this component, it's big enough

            # Reassign each point in the small component
            for local_i in comp_indices:
                orig_i = cls_mask[local_i]

                # Find 10 nearest neighbors in the full point cloud
                _, neighbor_indices = global_tree.query(coord[orig_i], k=11)

                # Pick the first neighbor label that isn't the same class
                reassigned = False
                for ni in neighbor_indices[1:]:  # skip self
                    if labels[ni] != cls:
                        filtered[orig_i] = labels[ni]
                        reassigned = True
                        break

                if not reassigned:
                    filtered[orig_i] = 0  # fallback to Unclassified

            total_removed += len(comp_indices)

    print(f"  [SC] Total removed: {total_removed:,} points across all rare classes")
    return filtered


# ==============================================================================
# MAIN PROCESSING
# ==============================================================================
def process_file(file_path, output_dir, args):
    stem     = Path(file_path).stem
    out_path = os.path.join(output_dir, f"PP_{stem}.laz")

    if os.path.exists(out_path):
        print(f"  [SKIP] Already exists: {out_path}")
        return

    print(f"\n{'='*60}")
    print(f"Processing: {Path(file_path).name}")
    print(f"{'='*60}")

    # ── Load LAZ ──────────────────────────────────────────────────
    try:
        las = laspy.read(file_path)
    except Exception as e:
        print(f"  [ERROR] Cannot read file: {e}")
        return

    x = np.array(las.x, dtype=np.float32)
    y = np.array(las.y, dtype=np.float32)
    z = np.array(las.z, dtype=np.float32)
    labels = np.array(las.classification, dtype=np.int32)
    n_total = len(x)

    print(f"  Points: {n_total:,}")

    # Print before stats
    print(f"\n  Class distribution BEFORE post-processing:")
    print(f"  {'ID':<5} {'Class':<15} {'Count':<12} {'%'}")
    print(f"  {'-'*45}")
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  {u:<5} {CLASS_NAMES.get(int(u), 'Unknown'):<15} {c:<12,} {c/n_total*100:.2f}%")

    # ── Build coordinate array ─────────────────────────────────────
    coord = np.stack([x, y, z], axis=1)

    # ── Filter 1: Majority Vote ────────────────────────────────────
    if not args.skip_mv:
        print(f"\n  Running Filter 1: Majority Vote...")
        labels = majority_vote_filter(
            coord, labels,
            radius=args.mv_radius,
            min_neighbors=args.mv_min_neighbors
        )
    else:
        print(f"\n  [SKIP] Majority Vote filter skipped.")

    # ── Filter 2: Small Component Removal ─────────────────────────
    if not args.skip_sc:
        print(f"\n  Running Filter 2: Small Component Removal...")
        labels = remove_small_components(
            coord, labels,
            min_points=args.sc_min_points,
            radius=args.sc_radius
        )
    else:
        print(f"\n  [SKIP] Small Component filter skipped.")

    # Print after stats
    print(f"\n  Class distribution AFTER post-processing:")
    print(f"  {'ID':<5} {'Class':<15} {'Count':<12} {'%'}")
    print(f"  {'-'*45}")
    unique, counts = np.unique(labels, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  {u:<5} {CLASS_NAMES.get(int(u), 'Unknown'):<15} {c:<12,} {c/n_total*100:.2f}%")

    # ── Save output LAZ ───────────────────────────────────────────
    header         = laspy.LasHeader(point_format=las.header.point_format,
                                     version=las.header.version)
    header.scales  = las.header.scales
    header.offsets = las.header.offsets
    out_las        = laspy.LasData(header=header)

    for dim in las.point_format.dimension_names:
        try:
            setattr(out_las, dim, getattr(las, dim))
        except Exception:
            pass

    out_las.classification = labels.astype(np.uint8)
    out_las.write(out_path)
    print(f"\n  ✓ Saved → {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Post-process predicted LAZ files with majority vote + component filtering"
    )
    parser.add_argument("--input_dir",          required=True,
                        help="Directory containing predicted LAZ files")
    parser.add_argument("--output_dir",         required=True,
                        help="Directory to save post-processed LAZ files")

    # Majority vote params
    parser.add_argument("--mv_radius",          type=float, default=0.30,
                        help="Majority vote search radius in metres (default: 0.15)")
    parser.add_argument("--mv_min_neighbors",   type=int,   default=50,
                        help="Min neighbors to trigger reassignment (default: 5)")
    parser.add_argument("--skip_mv",            action="store_true",
                        help="Skip majority vote filter")

    # Small component params
    parser.add_argument("--sc_min_points",      type=int,   default=50,
                        help="Min points to keep a component (default: 20)")
    parser.add_argument("--sc_radius",          type=float, default=0.50,
                        help="Connection radius for components in metres (default: 0.20)")
    parser.add_argument("--skip_sc",            action="store_true",
                        help="Skip small component filter")

    # Single file mode
    parser.add_argument("--single_file",        type=str,   default=None,
                        help="Process a single file instead of whole directory")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.single_file:
        # Single file mode — great for testing params on one file first
        process_file(args.single_file, args.output_dir, args)
    else:
        files = sorted([
            os.path.join(args.input_dir, f)
            for f in os.listdir(args.input_dir)
            if f.lower().endswith(".laz") or f.lower().endswith(".las")
        ])
        if not files:
            print(f"No LAZ files found in {args.input_dir}")
            return
        print(f"Found {len(files)} file(s) to process.\n")
        for fp in files:
            process_file(fp, args.output_dir, args)

    print(f"\n{'='*60}")
    print("All done.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()