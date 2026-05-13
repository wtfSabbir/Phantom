"""
predict_laz_iterative.py - PTv3 V2 self-feeding iterative inference with TTA.

No V1 model needed. Uses V2 model only.

How it works:
    Iteration 1: Feed zeros as v1_pred channel → rough predictions
    Iteration 2: Feed iteration 1 predictions as v1_pred → better predictions
    Iteration 3: Feed iteration 2 predictions as v1_pred → good predictions
    Final pass:  TTA on top of last iteration predictions → best predictions

This mimics the V1→V2 pipeline but uses only V2 weights.
The model bootstraps itself — each pass corrects the previous pass's mistakes.

TTA modes:
    --tta_mode none    →  no TTA on final pass (fastest)
    --tta_mode fast    →  4 augmentations on final pass
    --tta_mode normal  →  8 augmentations on final pass  ← recommended
    --tta_mode full    →  16 augmentations on final pass

Iterations:
    --iterations 3     →  3 self-feeding passes before TTA (recommended)
    --iterations 1     →  single pass with zeros (weakest)
    --iterations 4     →  4 passes (diminishing returns after 3)

Usage:

    # Single file test first
    python predict_laz_iterative.py \
        --single_file /path/to/input.laz \
        --output_dir  /path/to/output \
        --weight      /path/to/v2_model_best.pth \
        --iterations  3 \
        --tta_mode    normal

    # Full directory
    python predict_laz_iterative.py \
        --input_dir  /path/to/laz_files \
        --output_dir /path/to/output \
        --weight     /path/to/v2_model_best.pth \
        --iterations 3 \
        --tta_mode   normal
"""

import os
import argparse
import numpy as np
import torch
import laspy
from pathlib import Path
from tqdm import tqdm
from pointcept.models import build_model


# ==============================================================================
# CLASS DEFINITIONS — must match your training config
# ==============================================================================
TRAIN_TO_ORIGINAL = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:9, 9:10}
CLASS_NAMES = {
    0:  "Unclassified",
    1:  "Ground",
    2:  "Pole",
    3:  "Sign",
    4:  "Bollard",
    5:  "Trunk",
    6:  "Vegetation",
    7:  "Building",
    9:  "Fence",
    10: "Gate",
}
NUM_CLASSES  = 10
V2_MAX_CLASS = 9.0   # for normalizing predictions to [0, 1]


# ==============================================================================
# TTA AUGMENTATIONS
# ==============================================================================
def get_tta_augmentations(mode="normal"):
    """
    Returns list of augmentation dicts.
    First entry is always identity = plain single pass.
    mode='none' returns only identity = effectively no TTA.
    """
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
        raise ValueError(f"Unknown tta_mode '{mode}'. Choose: none / fast / normal / full")


def apply_augmentation(coord, aug):
    """
    Apply geometric augmentation to (N, 3) coordinate array.
    Returns augmented copy — original never modified.
    Intensity and v1_pred are NOT augmented.
    """
    c = coord.copy()

    if aug["flip_x"]:
        c[:, 0] *= -1.0

    if aug["flip_y"]:
        c[:, 1] *= -1.0

    angle_deg = aug["rotate_z"]
    if angle_deg != 0.0:
        angle_rad = np.deg2rad(angle_deg)
        cos_a   = np.cos(angle_rad)
        sin_a   = np.sin(angle_rad)
        x_new   = cos_a * c[:, 0] - sin_a * c[:, 1]
        y_new   = sin_a * c[:, 0] + cos_a * c[:, 1]
        c[:, 0] = x_new
        c[:, 1] = y_new

    return c


# ==============================================================================
# MODEL
# ==============================================================================
def build_v2_model(weight_path, device):
    """
    V2 model — 5 channels: XYZ(3) + intensity(1) + v1_pred_norm(1)
    Must match your V2 training config exactly.
    """
    model_cfg = dict(
        type="DefaultSegmentorV2",
        num_classes=10,
        backbone_out_channels=64,
        backbone=dict(
            type="PT-v3m1",
            in_channels=5,              # XYZ + intensity + v1_pred_norm
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
            #layer_scale=None,
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
    print(f"  V2 model loaded: {weight_path}")
    return model


# ==============================================================================
# VOXELIZATION
# ==============================================================================
def voxelize(coord, feat, grid_size=0.05):
    """
    Grid voxelization — one representative point per voxel.
    Returns voxelized coord, feat, grid_coord, and orig2vox mapping.
    """
    scaled   = coord / grid_size
    grid     = np.floor(scaled).astype(np.int64)
    grid    -= grid.min(axis=0)
    #key      = grid[:, 0] * 1_000_003 + grid[:, 1] * 1_000_033 + grid[:, 2]
    key = (grid[:, 0].astype(np.int64) << 40) | (grid[:, 1].astype(np.int64) << 20) | grid[:, 2].astype(np.int64)
    idx_sort = np.argsort(key)
    _, inverse, count = np.unique(
        key[idx_sort], return_inverse=True, return_counts=True
    )
    idx_sel  = (
        np.cumsum(np.insert(count, 0, 0)[:-1])
        + np.random.randint(0, count.max(), count.size) % count
    )
    idx_uniq = idx_sort[idx_sel]
    orig2vox = np.zeros(len(coord), dtype=np.int64)
    orig2vox[idx_sort] = inverse

    return coord[idx_uniq], feat[idx_uniq], grid[idx_uniq], orig2vox


# ==============================================================================
# SINGLE FORWARD PASS
# ==============================================================================
@torch.no_grad()
def forward_pass(model, coord_aug, feat, grid_size, device):
    """
    Run one forward pass.
    Returns softmax probabilities (N_original, NUM_CLASSES) as numpy.

    coord_aug: (N, 3) augmented + centred coordinates
    feat:      (N, 2) intensity + v1_pred_norm — NOT augmented
    """
    v_coord, v_feat, v_grid, orig2vox = voxelize(coord_aug, feat, grid_size)

    # Full feature = coord(3) + feat(2) = 5 channels
    full_feat = np.concatenate([v_coord, v_feat], axis=1).astype(np.float32)

    input_dict = dict(
        coord      = torch.from_numpy(v_coord).float().to(device),
        feat       = torch.from_numpy(full_feat).float().to(device),
        grid_coord = torch.from_numpy(v_grid).int().to(device),
        offset     = torch.tensor([len(v_coord)],
                                  dtype=torch.int32).to(device),
    )

    out    = model(input_dict)
    logits = out["seg_logits"] if isinstance(out, dict) else out
    probs  = torch.softmax(logits, dim=1).cpu().numpy()  # (N_vox, C)

    torch.cuda.empty_cache()
    return probs[orig2vox]   # (N_original, C)


# ==============================================================================
# ITERATIVE SELF-FEEDING INFERENCE
# ==============================================================================
def predict_tile_iterative(model, coord, intensity, grid_size,
                           device, n_iterations, augmentations):
    """
    Iterative self-feeding inference on one tile.

    Steps:
        For each iteration (except last):
            - Single forward pass
            - Use argmax prediction as next v1_pred

        Final iteration:
            - TTA (multiple augmented passes)
            - Average softmax probs
            - Argmax → final prediction

    coord:         (N, 3) centred coordinates
    intensity:     (N, 1) normalized intensity
    n_iterations:  number of self-feeding passes before TTA
    augmentations: list of aug dicts — len=1 means no TTA
    """
    n_points = len(coord)

    # ── Iterations 1 to N-1: single pass, no TTA ──────────────────
    # Start with zeros as v1_pred — model has no hint yet
    v1_pred_norm = np.zeros((n_points, 1), dtype=np.float32)

    for iteration in range(1, n_iterations + 1):

        # Build feature: intensity + v1_pred_norm
        feat = np.concatenate([intensity, v1_pred_norm], axis=1)  # (N, 2)

        is_final = (iteration == n_iterations)

        if is_final:
            # ── Final pass: run TTA ────────────────────────────────
            prob_sum = np.zeros((n_points, NUM_CLASSES), dtype=np.float32)

            for aug_idx, aug in enumerate(augmentations):
                coord_aug = apply_augmentation(coord, aug)
                probs     = forward_pass(model, coord_aug, feat,
                                         grid_size, device)
                prob_sum += probs

            prob_avg    = prob_sum / len(augmentations)
            pred        = np.argmax(prob_avg, axis=1)  # (N,) final prediction

        else:
            # ── Intermediate pass: single forward, identity aug ────
            probs = forward_pass(model, coord, feat, grid_size, device)
            pred  = np.argmax(probs, axis=1)  # (N,) rough prediction

            # Feed prediction back as next v1_pred
            # Normalize exactly as LidarDatasetV2: pred / 9.0
            v1_pred_norm = (pred.astype(np.float32) / V2_MAX_CLASS).reshape(-1, 1)

        n_tta = len(augmentations) if is_final else 1
        print(f"    Iter {iteration}/{n_iterations}"
              f"{'  [TTA x' + str(n_tta) + ']' if is_final and n_tta > 1 else '  [single]'}"
              f"  unique_preds={len(np.unique(pred))}")

    return pred


# ==============================================================================
# PROCESS ONE FILE
# ==============================================================================
def predict_file(model, file_path, output_dir, args, augmentations, device):
    stem     = Path(file_path).stem
    out_path = os.path.join(output_dir, f"ITER_{stem}.laz")

    if os.path.exists(out_path):
        print(f"  [SKIP] Already exists: {out_path}")
        return

    print(f"\n{'='*60}")
    print(f"File:        {Path(file_path).name}")
    print(f"Iterations:  {args.iterations}")
    print(f"TTA mode:    {args.tta_mode} ({len(augmentations)} passes on final iter)")
    print(f"Total fwd:   {args.iterations - 1} single + {len(augmentations)} TTA"
          f" = {args.iterations - 1 + len(augmentations)} forward passes")
    print(f"{'='*60}")

    # ── Load LAZ ──────────────────────────────────────────────────
    try:
        las = laspy.read(file_path)
    except Exception as e:
        print(f"  [ERROR] Cannot read: {e}")
        return

    x = np.array(las.x, dtype=np.float64)
    y = np.array(las.y, dtype=np.float64)
    z = np.array(las.z, dtype=np.float64)
    n_total = len(x)

    intensity = (
        np.array(las.intensity, dtype=np.float32) / 65535.0
        if hasattr(las, "intensity")
        else np.zeros(n_total, dtype=np.float32)
    )
    intensity = intensity.reshape(-1, 1)   # (N, 1)

    print(f"  Total points: {n_total:,}")

    # Global mean — consistent centring across all tiles
    global_mean = np.array(
        [x.mean(), y.mean(), z.mean()], dtype=np.float32
    )

    x_range = x.max() - x.min()
    y_range = y.max() - y.min()
    needs_chunking = (x_range > args.tile_threshold) or \
                     (y_range > args.tile_threshold)

    if needs_chunking:
        x_starts = np.arange(x.min(), x.max(), args.tile_size)
        y_starts = np.arange(y.min(), y.max(), args.tile_size)
        print(f"  Scene: {x_range:.0f}x{y_range:.0f}m "
              f"→ {len(x_starts)*len(y_starts)} tiles of {args.tile_size}m")
    else:
        x_starts    = [x.min()]
        y_starts    = [y.min()]
        tile_size_x = x_range + 1.0
        tile_size_y = y_range + 1.0
        print(f"  Scene: {x_range:.0f}x{y_range:.0f}m → single tile")

    pred_all = np.zeros(n_total, dtype=np.int64)

    pbar = tqdm(
        total=len(x_starts) * len(y_starts),
        desc="  Tiles",
        leave=False
    )

    for x0 in x_starts:
        for y0 in y_starts:
            x1 = x0 + (args.tile_size if needs_chunking else tile_size_x)
            y1 = y0 + (args.tile_size if needs_chunking else tile_size_y)

            mask = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
            pbar.update(1)
            if mask.sum() < 100:
                continue

            coord = np.stack([
                x[mask].astype(np.float32),
                y[mask].astype(np.float32),
                z[mask].astype(np.float32),
            ], axis=1)
            coord -= global_mean

            tile_intensity = intensity[mask]

            try:
                pred_tile = predict_tile_iterative(
                    model, coord, tile_intensity,
                    args.grid_size, device,
                    args.iterations, augmentations
                )
                pred_all[mask] = pred_tile

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"\n  [OOM] Try --tile_size 15 or --tta_mode fast")
                else:
                    print(f"\n  [WARN] Tile failed: {e}")
                torch.cuda.empty_cache()

    pbar.close()

    # ── Remap train IDs → original LAZ class IDs ──────────────────
    pred_original = np.zeros_like(pred_all)
    for train_id, orig_id in TRAIN_TO_ORIGINAL.items():
        pred_original[pred_all == train_id] = orig_id

    # ── Class distribution ─────────────────────────────────────────
    print(f"\n  Class distribution:")
    print(f"  {'ID':<5} {'Class':<15} {'Count':<12} {'%'}")
    print(f"  {'-'*45}")
    unique, counts = np.unique(pred_original, return_counts=True)
    for u, c in zip(unique, counts):
        name = CLASS_NAMES.get(int(u), f"Unknown({u})")
        print(f"  {u:<5} {name:<15} {c:<12,} {c/n_total*100:.1f}%")

    # ── Save LAZ ───────────────────────────────────────────────────
    header        = laspy.LasHeader(
                        point_format=las.header.point_format,
                        version=las.header.version
                    )
    header.scales  = las.header.scales
    header.offsets = las.header.offsets
    out_las        = laspy.LasData(header=header)

    for dim in las.point_format.dimension_names:
        try:
            setattr(out_las, dim, getattr(las, dim))
        except Exception:
            pass

    out_las.classification = pred_original.astype(np.uint8)
    out_las.write(out_path)
    print(f"\n  ✓ Saved → {out_path}")


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="PTv3 V2 iterative self-feeding inference with TTA — no V1 needed"
    )

    # Required
    parser.add_argument("--output_dir",     required=True)
    parser.add_argument("--weight",         required=True,
                        help="Path to V2 model checkpoint (5ch)")

    # Input
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input_dir",   type=str,
                       help="Directory of LAZ files")
    group.add_argument("--single_file", type=str,
                       help="Single LAZ file for testing")

    # Iteration control
    parser.add_argument("--iterations",     type=int,   default=3,
                        help="Number of self-feeding passes (default: 3)")

    # TTA on final pass
    parser.add_argument("--tta_mode",       type=str,   default="normal",
                        choices=["none", "fast", "normal", "full"],
                        help="TTA on final pass: none/fast/normal/full (default: normal)")

    # Tiling
    parser.add_argument("--tile_size",      type=float, default=25.0)
    parser.add_argument("--tile_threshold", type=float, default=30.0)

    # Voxelization — must match training
    parser.add_argument("--grid_size",      type=float, default=0.05)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    model = build_v2_model(args.weight, device)

    augmentations = get_tta_augmentations(args.tta_mode)

    print(f"Iterations:  {args.iterations}")
    print(f"TTA mode:    {args.tta_mode} ({len(augmentations)} passes on final iter)")
    print(f"Total fwd passes per tile: "
          f"{args.iterations - 1} single + {len(augmentations)} TTA"
          f" = {args.iterations - 1 + len(augmentations)}")
    print(f"Grid size:   {args.grid_size}m\n")

    if args.single_file:
        predict_file(model, args.single_file, args.output_dir,
                     args, augmentations, device)
    else:
        files = sorted([
            os.path.join(args.input_dir, f)
            for f in os.listdir(args.input_dir)
            if f.lower().endswith(".laz") or f.lower().endswith(".las")
        ])
        if not files:
            print(f"No LAZ files found in {args.input_dir}")
            return
        print(f"Found {len(files)} file(s).\n")
        for fp in files:
            predict_file(model, fp, args.output_dir,
                         args, augmentations, device)

    print(f"\n{'='*60}")
    print("All done.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()