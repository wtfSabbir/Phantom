"""
active_learning_score.py - Active Learning uncertainty scoring for unlabeled LAZ files.

What this does:
    Runs inference on a folder of UNLABELED LAZ files.
    For each file, measures how uncertain the model is about its predictions.
    Produces a ranked CSV — most uncertain files first.
    These are the files you should annotate first.

    Also saves the predicted LAZ with an extra uncertainty field so you can
    visualize WHERE in each scene the model is confused (open in CloudCompare,
    color by 'uncertainty' scalar field).

Why this matters:
    Not all unlabeled files are equally useful to annotate.
    Files where the model is confident → annotating them teaches nothing new.
    Files where the model is confused  → annotating them improves the model most.
    This script finds those confused files so you spend annotation budget wisely.

Uncertainty method:
    Class-weighted entropy per point.
    Rare/hard classes (Gate, Sign, Bollard, Pole, Fence, Trunk) weighted higher.
    Easy classes (Vegetation, Building, Ground) weighted lower.
    Scene score = mean weighted entropy across all points.
    Higher score = more uncertain = higher annotation priority.

Output:
    <output_dir>/
        uncertainty_ranking.csv         ← ranked list of all files with scores
        predicted/
            PRED_tile_001.laz           ← predicted LAZ with uncertainty field
            PRED_tile_002.laz
            ...

CSV columns:
    rank                  : annotation priority (1 = annotate first)
    filename              : LAZ file name
    uncertainty_score     : overall weighted uncertainty score (higher = more uncertain)
    top_uncertain_class   : class the model was most uncertain about
    n_points              : total points in file
    <class>_uncertainty   : per-class uncertainty contribution (one column per class)

Usage:

    # Score all files in a directory
    python active_learning_score.py \
        --input_dir  /path/to/unlabeled_laz \
        --output_dir /path/to/active_learning_output \
        --weight     /path/to/v2_model_best.pth \
        --top_k      30

    # Score a single file (for testing)
    python active_learning_score.py \
        --single_file /path/to/test.laz \
        --output_dir  /path/to/active_learning_output \
        --weight      /path/to/v2_model_best.pth

    # Skip saving predicted LAZ (faster, CSV only)
    python active_learning_score.py \
        --input_dir  /path/to/unlabeled_laz \
        --output_dir /path/to/active_learning_output \
        --weight     /path/to/v2_model_best.pth \
        --no_save_laz
"""

import os
import argparse
import csv
import shutil
import numpy as np
import torch
import laspy
from pathlib import Path
from tqdm import tqdm

from pointcept.models import build_model


# ==============================================================================
# CLASS DEFINITIONS — must match your training config
# ==============================================================================
CLASS_NAMES = {
    0: "unclassified",
    1: "ground",
    2: "pole",
    3: "sign",
    4: "bollard",
    5: "trunk",
    6: "vegetation",
    7: "building",
    8: "fence",
    9: "gate",
}
TRAIN_TO_ORIGINAL = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:9, 9:10}
NUM_CLASSES  = 10
V2_MAX_CLASS = 9.0

# Class weights for uncertainty scoring
# Higher = model uncertainty about this class matters more for ranking
# Rare/struggling classes get high weight so files with uncertain rare
# class predictions get ranked higher for annotation
CLASS_UNCERTAINTY_WEIGHTS = np.array([
    0.2,   # 0 unclassified  — don't prioritize
    0.2,   # 1 ground        — easy, dominant
    2.0,   # 2 pole          — rare, struggling
    3.0,   # 3 sign          — very rare, worst performers
    3.0,   # 4 bollard       — very rare
    2.0,   # 5 trunk         — rare
    0.2,   # 6 vegetation    — easy, dominant
    0.2,   # 7 building      — easy, dominant
    2.0,   # 8 fence         — rare
    3.0,   # 9 gate          — worst performing class
], dtype=np.float32)


# ==============================================================================
# MODEL
# ==============================================================================
def build_v2_model(weight_path, device):
    """
    Build and load V2 model.
    Architecture must match your training config exactly.
    """
    model_cfg = dict(
        type="DefaultSegmentorV2",
        num_classes=10,
        backbone_out_channels=64,
        backbone=dict(
            type="PT-v3m1",
            in_channels=5,
            order=("z", "z-trans", "hilbert", "hilbert-trans"),
            stride=(2, 2, 2, 2),
            enc_depths=(2, 2, 2, 6, 2),
            enc_channels=(32, 64, 128, 256, 512),
            enc_num_head=(2, 4, 8, 16, 32),
            enc_patch_size=(48, 48, 48, 48, 48),
            dec_depths=(2, 2, 2, 2),
            dec_channels=(64, 64, 128, 256),
            dec_num_head=(4, 4, 8, 16),
            dec_patch_size=(48, 48, 48, 48),
            mlp_ratio=4, qkv_bias=True, qk_scale=None,
            attn_drop=0.0, proj_drop=0.0, drop_path=0.3,
            shuffle_orders=True, pre_norm=True,
            enable_rpe=False, enable_flash=False,
            upcast_attention=False, upcast_softmax=False,
        ),
        criteria=[
            dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=255)
        ],
    )
    model = build_model(model_cfg)
    ckpt  = torch.load(weight_path, map_location=device, weights_only=False)
    sd    = ckpt.get("state_dict", ckpt.get("model", ckpt))
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    print(f"Model loaded: {weight_path}")
    return model


# ==============================================================================
# VOXELIZATION
# ==============================================================================
def voxelize(coord, feat, grid_size=0.04):
    scaled   = coord / grid_size
    grid     = np.floor(scaled).astype(np.int64)
    grid    -= grid.min(axis=0)
    key      = (
        (grid[:, 0].astype(np.int64) << 40) |
        (grid[:, 1].astype(np.int64) << 20) |
         grid[:, 2].astype(np.int64)
    )
    idx_sort = np.argsort(key)
    _, inverse, count = np.unique(
        key[idx_sort], return_inverse=True, return_counts=True
    )
    idx_sel = (
        np.cumsum(np.insert(count, 0, 0)[:-1])
        + np.random.randint(0, count.max(), count.size) % count
    )
    idx_uniq = idx_sort[idx_sel]
    orig2vox = np.zeros(len(coord), dtype=np.int64)
    orig2vox[idx_sort] = inverse
    return coord[idx_uniq], feat[idx_uniq], grid[idx_uniq], orig2vox


# ==============================================================================
# SINGLE FORWARD PASS — returns PROBABILITIES not just argmax
# ==============================================================================
@torch.no_grad()
def forward_pass(model, coord_aug, feat, grid_size, device):
    """
    Run one forward pass.
    Returns softmax probabilities (N_original, NUM_CLASSES).
    We need the full probability distribution for uncertainty calculation,
    not just the argmax prediction.
    """
    v_coord, v_feat, v_grid, orig2vox = voxelize(coord_aug, feat, grid_size)
    full_feat = np.concatenate([v_coord, v_feat], axis=1).astype(np.float32)

    input_dict = dict(
        coord      = torch.from_numpy(v_coord).float().to(device),
        feat       = torch.from_numpy(full_feat).float().to(device),
        grid_coord = torch.from_numpy(v_grid).int().to(device),
        offset     = torch.tensor([len(v_coord)], dtype=torch.int32).to(device),
    )

    out    = model(input_dict)
    logits = out["seg_logits"] if isinstance(out, dict) else out
    probs  = torch.softmax(logits, dim=1).cpu().numpy()   # (N_vox, C)

    torch.cuda.empty_cache()
    return probs[orig2vox]   # (N_original, C)


# ==============================================================================
# SELF-FEEDING INFERENCE — returns averaged probabilities
# ==============================================================================
def predict_tile_with_uncertainty(model, coord, intensity, grid_size,
                                   device, n_iterations, augmentations):
    """
    Run self-feeding iterative inference and return AVERAGED softmax probs.
    We return the full probability distribution so we can compute uncertainty.

    Returns:
        probs : (N, NUM_CLASSES) averaged softmax probabilities
        pred  : (N,) argmax predictions
    """
    n_points     = len(coord)
    v1_pred_norm = np.zeros((n_points, 1), dtype=np.float32)

    for iteration in range(1, n_iterations + 1):
        feat     = np.concatenate([intensity, v1_pred_norm], axis=1)
        is_final = (iteration == n_iterations)

        if is_final:
            # Final pass — average over TTA augmentations
            prob_sum = np.zeros((n_points, NUM_CLASSES), dtype=np.float32)
            for aug in augmentations:
                coord_aug = apply_augmentation(coord, aug)
                prob_sum += forward_pass(model, coord_aug, feat, grid_size, device)
            probs = prob_sum / len(augmentations)
            pred  = np.argmax(probs, axis=1)
        else:
            # Intermediate pass — single forward, feed back prediction
            probs = forward_pass(model, coord, feat, grid_size, device)
            pred  = np.argmax(probs, axis=1)
            v1_pred_norm = (
                pred.astype(np.float32) / V2_MAX_CLASS
            ).reshape(-1, 1)

    return probs, pred


# ==============================================================================
# UNCERTAINTY CALCULATION
# ==============================================================================
def compute_uncertainty(probs):
    """
    Compute per-point uncertainty using class-weighted entropy.

    probs : (N, NUM_CLASSES) softmax probabilities

    Returns:
        point_uncertainty : (N,) per-point uncertainty score [0, 1]
        scene_score       : float  overall scene uncertainty score
        class_scores      : dict   per-class contribution to scene score
        top_class         : str    class name with highest uncertainty contribution
    """
    # Clip for numerical stability
    probs_clipped = np.clip(probs, 1e-10, 1.0)

    # Per-point entropy: -sum(p * log(p)) for each point
    # High entropy = probability spread across many classes = confused
    entropy = -np.sum(probs_clipped * np.log(probs_clipped), axis=1)  # (N,)

    # Normalize entropy to [0, 1] range
    # Max entropy = log(NUM_CLASSES) when all classes equally likely
    max_entropy = np.log(NUM_CLASSES)
    entropy_norm = entropy / max_entropy   # (N,)

    # Per-point weighted uncertainty
    # Weight each point's entropy by the weight of its PREDICTED class
    predicted_classes   = np.argmax(probs, axis=1)  # (N,)
    point_class_weights = CLASS_UNCERTAINTY_WEIGHTS[predicted_classes]  # (N,)
    point_uncertainty   = entropy_norm * point_class_weights   # (N,)

    # Scene score = mean weighted uncertainty across all points
    scene_score = float(np.mean(point_uncertainty))

    # Per-class contribution to scene score
    # For each class: mean uncertainty of points predicted as that class
    class_scores = {}
    for cls_id, cls_name in CLASS_NAMES.items():
        cls_mask = predicted_classes == cls_id
        if cls_mask.sum() > 0:
            class_scores[cls_name] = float(np.mean(point_uncertainty[cls_mask]))
        else:
            class_scores[cls_name] = 0.0

    # Which class contributes most to overall uncertainty
    # Weight by class importance to find the most problematic class
    weighted_class_scores = {
        cls_name: class_scores[cls_name] * CLASS_UNCERTAINTY_WEIGHTS[cls_id]
        for cls_id, cls_name in CLASS_NAMES.items()
    }
    top_class = max(weighted_class_scores, key=weighted_class_scores.get)

    return point_uncertainty, scene_score, class_scores, top_class


# ==============================================================================
# TTA AUGMENTATIONS
# ==============================================================================
def get_tta_augmentations(mode="normal"):
    identity = dict(rotate_z=0.0, flip_x=False, flip_y=False)
    if mode == "none":
        return [identity]
    elif mode == "fast":
        return [
            identity,
            dict(rotate_z=90.0,  flip_x=False, flip_y=False),
            dict(rotate_z=180.0, flip_x=False, flip_y=False),
            dict(rotate_z=270.0, flip_x=False, flip_y=False),
        ]
    elif mode == "normal":
        return [
            identity,
            dict(rotate_z=90.0,  flip_x=False, flip_y=False),
            dict(rotate_z=180.0, flip_x=False, flip_y=False),
            dict(rotate_z=270.0, flip_x=False, flip_y=False),
            dict(rotate_z=0.0,   flip_x=True,  flip_y=False),
            dict(rotate_z=90.0,  flip_x=True,  flip_y=False),
            dict(rotate_z=0.0,   flip_x=False, flip_y=True),
            dict(rotate_z=90.0,  flip_x=False, flip_y=True),
        ]
    elif mode == "full":
        return [
            identity,
            dict(rotate_z=45.0,  flip_x=False, flip_y=False),
            dict(rotate_z=90.0,  flip_x=False, flip_y=False),
            dict(rotate_z=135.0, flip_x=False, flip_y=False),
            dict(rotate_z=180.0, flip_x=False, flip_y=False),
            dict(rotate_z=225.0, flip_x=False, flip_y=False),
            dict(rotate_z=270.0, flip_x=False, flip_y=False),
            dict(rotate_z=315.0, flip_x=False, flip_y=False),
            dict(rotate_z=0.0,   flip_x=True,  flip_y=False),
            dict(rotate_z=90.0,  flip_x=True,  flip_y=False),
            dict(rotate_z=180.0, flip_x=True,  flip_y=False),
            dict(rotate_z=270.0, flip_x=True,  flip_y=False),
            dict(rotate_z=0.0,   flip_x=False, flip_y=True),
            dict(rotate_z=90.0,  flip_x=False, flip_y=True),
            dict(rotate_z=0.0,   flip_x=True,  flip_y=True),
            dict(rotate_z=90.0,  flip_x=True,  flip_y=True),
        ]
    else:
        raise ValueError(f"Unknown tta_mode '{mode}'.")


def apply_augmentation(coord, aug):
    c = coord.copy()
    if aug["flip_x"]:
        c[:, 0] *= -1.0
    if aug["flip_y"]:
        c[:, 1] *= -1.0
    angle_deg = aug["rotate_z"]
    if angle_deg != 0.0:
        angle_rad = np.deg2rad(angle_deg)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        x_new = cos_a * c[:, 0] - sin_a * c[:, 1]
        y_new = sin_a * c[:, 0] + cos_a * c[:, 1]
        c[:, 0] = x_new
        c[:, 1] = y_new
    return c


# ==============================================================================
# PROCESS ONE FILE
# ==============================================================================
def process_file(model, file_path, pred_output_dir, args, augmentations, device):
    """
    Run inference on one file.
    Returns a result dict with uncertainty scores, or None if failed.
    """
    print(f"\n  Processing: {Path(file_path).name}")

    try:
        las = laspy.read(file_path)
    except Exception as e:
        print(f"  [ERROR] Cannot read: {e}")
        return None

    x = np.array(las.x, dtype=np.float64)
    y = np.array(las.y, dtype=np.float64)
    z = np.array(las.z, dtype=np.float64)
    n_total = len(x)

    intensity = (
        np.array(las.intensity, dtype=np.float32) / 65535.0
        if hasattr(las, "intensity")
        else np.zeros(n_total, dtype=np.float32)
    )
    intensity = intensity.reshape(-1, 1)

    global_z_ground = np.percentile(z, 1)

    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    needs_chunking = (
        x_range > args.tile_threshold or y_range > args.tile_threshold
    )

    if needs_chunking:
        x_starts = np.arange(x.min(), x.max(), args.tile_size)
        y_starts = np.arange(y.min(), y.max(), args.tile_size)
    else:
        x_starts    = [x.min()]
        y_starts    = [y.min()]
        tile_size_x = x_range + 1.0
        tile_size_y = y_range + 1.0

    # Arrays to collect results across all tiles
    pred_all        = np.zeros(n_total, dtype=np.int64)
    uncertainty_all = np.zeros(n_total, dtype=np.float32)

    for x0 in x_starts:
        for y0 in y_starts:
            x1 = x0 + (args.tile_size if needs_chunking else tile_size_x)
            y1 = y0 + (args.tile_size if needs_chunking else tile_size_y)

            margin     = 2.0
            inner_mask = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
            outer_mask = (
                (x >= x0 - margin) & (x < x1 + margin) &
                (y >= y0 - margin) & (y < y1 + margin)
            )

            if inner_mask.sum() < 100:
                continue

            tile_x = x[outer_mask].astype(np.float32)
            tile_y = y[outer_mask].astype(np.float32)
            tile_z = z[outer_mask].astype(np.float32)

            coord = np.stack([
                tile_x,
                tile_y,
                tile_z - global_z_ground,
            ], axis=1)
            coord[:, 0] -= tile_x.mean()
            coord[:, 1] -= tile_y.mean()

            try:
                # Get full probability distribution (not just argmax)
                probs, pred = predict_tile_with_uncertainty(
                    model, coord, intensity[outer_mask],
                    args.grid_size, device,
                    args.iterations, augmentations
                )

                # Compute per-point uncertainty
                point_uncertainty, _, _, _ = compute_uncertainty(probs)

                # Store only inner tile results
                is_inner = inner_mask[outer_mask]
                pred_all[inner_mask]        = pred[is_inner]
                uncertainty_all[inner_mask] = point_uncertainty[is_inner]

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  [OOM] Try --tile_size 15 or --tta_mode fast")
                else:
                    print(f"  [WARN] Tile failed: {e}")
                torch.cuda.empty_cache()

    # ── Compute scene-level uncertainty scores ────────────────────
    # Use the full probability distribution reconstructed from predictions
    # For scene scoring we use the stored uncertainty scores per point
    scene_score = float(np.mean(uncertainty_all))

    # Per-class uncertainty: mean uncertainty of points predicted as each class
    class_scores = {}
    top_class    = "unknown"
    top_score    = -1.0
    for cls_id, cls_name in CLASS_NAMES.items():
        cls_mask = pred_all == cls_id
        if cls_mask.sum() > 0:
            cls_score = float(
                np.mean(uncertainty_all[cls_mask]) *
                CLASS_UNCERTAINTY_WEIGHTS[cls_id]
            )
        else:
            cls_score = 0.0
        class_scores[cls_name] = cls_score
        if cls_score > top_score:
            top_score = cls_score
            top_class = cls_name

    print(f"  Points: {n_total:,}  |  Uncertainty score: {scene_score:.4f}"
          f"  |  Most uncertain class: {top_class}")

    # ── Remap train IDs → original LAZ class IDs ─────────────────
    pred_original = np.zeros_like(pred_all)
    for train_id, orig_id in TRAIN_TO_ORIGINAL.items():
        pred_original[pred_all == train_id] = orig_id

    # ── Save predicted LAZ with uncertainty field ─────────────────
    if not args.no_save_laz:
        stem     = Path(file_path).stem
        out_path = os.path.join(pred_output_dir, f"PRED_{stem}.laz")

        header        = laspy.LasHeader(
                            point_format=las.header.point_format,
                            version=las.header.version
                        )
        header.scales  = las.header.scales
        header.offsets = las.header.offsets

        # Add extra dimension for uncertainty score
        header.add_extra_dim(
            laspy.ExtraBytesParams(
                name="uncertainty",
                type=np.float32,
                description="Per-point model uncertainty [0=confident, 1=confused]"
            )
        )

        out_las = laspy.LasData(header=header)
        for dim in las.point_format.dimension_names:
            try:
                setattr(out_las, dim, getattr(las, dim))
            except Exception:
                pass

        out_las.classification = pred_original.astype(np.uint8)
        out_las.uncertainty    = uncertainty_all.astype(np.float32)
        out_las.write(out_path)
        print(f"  Saved → {out_path}")

    return {
        "filename"           : Path(file_path).name,
        "uncertainty_score"  : scene_score,
        "top_uncertain_class": top_class,
        "n_points"           : n_total,
        "class_scores"       : class_scores,
    }


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Active Learning uncertainty scoring for unlabeled LAZ files. "
            "Ranks files by how uncertain the model is — annotate top-ranked files first."
        )
    )

    # Input
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input_dir",   type=str,
                       help="Directory of unlabeled LAZ files")
    group.add_argument("--single_file", type=str,
                       help="Single LAZ file for testing")

    # Required
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save CSV ranking and predicted LAZ files")
    parser.add_argument("--weight",     required=True,
                        help="Path to V2 model checkpoint")

    # Optional
    parser.add_argument("--top_k",          type=int,   default=30,
                        help="Copy top-K most uncertain files to a separate "
                             "'annotate_these' folder (default: 30)")
    parser.add_argument("--tta_mode",        type=str,   default="fast",
                        choices=["none", "fast", "normal", "full"],
                        help="TTA mode (default: fast — quicker for scoring)")
    parser.add_argument("--iterations",      type=int,   default=3,
                        help="Self-feeding iterations (default: 3)")
    parser.add_argument("--grid_size",       type=float, default=0.04,
                        help="Voxel size — must match training (default: 0.04)")
    parser.add_argument("--tile_size",       type=float, default=25.0)
    parser.add_argument("--tile_threshold",  type=float, default=30.0)
    parser.add_argument("--no_save_laz",     action="store_true",
                        help="Skip saving predicted LAZ files (CSV only, faster)")

    args = parser.parse_args()

    # ── Setup output folders ──────────────────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    pred_output_dir = os.path.join(args.output_dir, "predicted")
    if not args.no_save_laz:
        os.makedirs(pred_output_dir, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model         = build_v2_model(args.weight, device)
    augmentations = get_tta_augmentations(args.tta_mode)

    print(f"TTA mode:   {args.tta_mode} ({len(augmentations)} passes)")
    print(f"Iterations: {args.iterations}")
    print(f"Grid size:  {args.grid_size}m")
    print(f"Top-K:      {args.top_k} files will be copied to 'annotate_these' folder\n")

    # ── Collect files ─────────────────────────────────────────────
    if args.single_file:
        files = [args.single_file]
    else:
        files = sorted([
            os.path.join(args.input_dir, f)
            for f in os.listdir(args.input_dir)
            if f.lower().endswith(".laz") or f.lower().endswith(".las")
        ])
        if not files:
            print(f"No LAZ files found in {args.input_dir}")
            return
    print(f"Found {len(files)} file(s) to score.\n")

    # ── Process all files ─────────────────────────────────────────
    results = []
    for fp in tqdm(files, desc="Scoring files"):
        result = process_file(
            model, fp, pred_output_dir, args, augmentations, device
        )
        if result is not None:
            results.append(result)

    if not results:
        print("No files processed successfully.")
        return

    # ── Rank by uncertainty score (highest first) ─────────────────
    results.sort(key=lambda r: r["uncertainty_score"], reverse=True)

    # ── Save CSV ranking ──────────────────────────────────────────
    csv_path = os.path.join(args.output_dir, "uncertainty_ranking.csv")

    fieldnames = (
        ["rank", "filename", "uncertainty_score", "top_uncertain_class", "n_points"]
        + [f"{cls}_uncertainty" for cls in CLASS_NAMES.values()]
    )

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rank, result in enumerate(results, start=1):
            row = {
                "rank"                : rank,
                "filename"            : result["filename"],
                "uncertainty_score"   : f"{result['uncertainty_score']:.4f}",
                "top_uncertain_class" : result["top_uncertain_class"],
                "n_points"            : result["n_points"],
            }
            for cls_name in CLASS_NAMES.values():
                row[f"{cls_name}_uncertainty"] = (
                    f"{result['class_scores'].get(cls_name, 0.0):.4f}"
                )
            writer.writerow(row)

    print(f"\n{'='*60}")
    print(f"Ranking saved → {csv_path}")
    print(f"{'='*60}")

    # ── Print top results ─────────────────────────────────────────
    print(f"\nTop {min(10, len(results))} most uncertain files "
          f"(annotate these first):\n")
    print(f"{'Rank':<6} {'Filename':<45} {'Score':<8} {'Most Uncertain Class'}")
    print("-" * 75)
    for r in results[:10]:
        print(
            f"{r['rank']:<6} "
            f"{r['filename']:<45} "
            f"{r['uncertainty_score']:.4f}   "
            f"{r['top_uncertain_class']}"
        )

    # ── Copy top-K files to annotation folder ─────────────────────
    if args.top_k > 0 and not args.single_file:
        annotate_dir = os.path.join(args.output_dir, "annotate_these")
        os.makedirs(annotate_dir, exist_ok=True)
        top_k = min(args.top_k, len(results))
        print(f"\nCopying top {top_k} files to: {annotate_dir}")
        for result in results[:top_k]:
            src = os.path.join(args.input_dir, result["filename"])
            dst = os.path.join(annotate_dir, result["filename"])
            if os.path.exists(src):
                shutil.copy2(src, dst)
        print(f"Done. Open these {top_k} files in CloudCompare and annotate them.")
        print(f"Tip: color by 'uncertainty' scalar field to see WHERE "
              f"the model is confused within each file.")

    print(f"\n{'='*60}")
    print("All done.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()