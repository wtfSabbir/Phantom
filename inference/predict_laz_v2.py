"""
predict_laz_v2.py - Production inference with v2 model.

Internally runs v1 first to generate v1_pred in memory,
then feeds [intensity, v1_pred] to v2 model.

From the outside: raw LAZ in -> predicted LAZ out. One command. Clean.

Usage:
    python predict_laz_v2.py \
        --input_dir  /mnt/d/PointCloudsFiles/Unclassified_Data \
        --output_dir /mnt/d/PointCloudsFiles/Predictions_V2 \
        --weight_v1  exp/laz_static/lidar_ptv3_run1/model/model_best.pth \
        --weight_v2  exp/laz_static/lidar_ptv3_v2/model/model_best.pth
"""

import os
import argparse
import numpy as np
import torch
import laspy
from pathlib import Path
from tqdm import tqdm
from pointcept.models import build_model

TRAIN_TO_ORIGINAL = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:9, 9:10}
CLASS_NAMES = {0:"Unclassified", 1:"Ground", 2:"Pole", 3:"Sign", 4:"Bollard",
               5:"Trunk", 6:"Vegetation", 7:"Building", 9:"Fence", 10:"Gate"}


def build_model_v1(weight_path, device):
    """Build v1 model - in_channels=4."""
    cfg = dict(
        type="DefaultSegmentorV2", num_classes=10, backbone_out_channels=64,
        backbone=dict(
            type="PT-v3m1", in_channels=4,
            order=("z","z-trans","hilbert","hilbert-trans"), stride=(2,2,2,2),
            enc_depths=(2,2,2,6,2), enc_channels=(32,64,128,256,512),
            enc_num_head=(2,4,8,16,32), enc_patch_size=(48,48,48,48,48),
            dec_depths=(2,2,2,2), dec_channels=(64,64,128,256),
            dec_num_head=(4,4,8,16), dec_patch_size=(48,48,48,48),
            mlp_ratio=4, qkv_bias=True, qk_scale=None,
            attn_drop=0.0, proj_drop=0.0, drop_path=0.3,
            shuffle_orders=True, pre_norm=True,
            enable_rpe=False, enable_flash=False,
            upcast_attention=False, upcast_softmax=False,
        ),
        criteria=[dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=255)],
    )
    model = build_model(cfg)
    ckpt  = torch.load(weight_path, map_location=device)
    sd    = ckpt.get("state_dict", ckpt.get("model", ckpt))
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    print(f"  V1 model loaded: {weight_path}")
    return model


def build_model_v2(weight_path, device):
    """Build v2 model - in_channels=5."""
    cfg = dict(
        type="DefaultSegmentorV2", num_classes=10, backbone_out_channels=64,
        backbone=dict(
            type="PT-v3m1", in_channels=5,
            order=("z","z-trans","hilbert","hilbert-trans"), stride=(2,2,2,2),
            enc_depths=(2,2,2,6,2), enc_channels=(32,64,128,256,512),
            enc_num_head=(2,4,8,16,32), enc_patch_size=(48,48,48,48,48),
            dec_depths=(2,2,2,2), dec_channels=(64,64,128,256),
            dec_num_head=(4,4,8,16), dec_patch_size=(48,48,48,48),
            mlp_ratio=4, qkv_bias=True, qk_scale=None,
            attn_drop=0.0, proj_drop=0.0, drop_path=0.3,
            shuffle_orders=True, pre_norm=True,
            enable_rpe=False, enable_flash=False,
            upcast_attention=False, upcast_softmax=False,
        ),
        criteria=[dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=255)],
    )
    model = build_model(cfg)
    ckpt  = torch.load(weight_path, map_location=device)
    sd    = ckpt.get("state_dict", ckpt.get("model", ckpt))
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    print(f"  V2 model loaded: {weight_path}")
    return model


def voxelize(coord, color, grid_size=0.05):
    scaled   = coord / grid_size
    grid     = np.floor(scaled).astype(np.int64)
    grid    -= grid.min(axis=0)
    key      = grid[:,0]*1000003 + grid[:,1]*1000033 + grid[:,2]
    idx_sort = np.argsort(key)
    _, inverse, count = np.unique(key[idx_sort], return_inverse=True, return_counts=True)
    idx_sel  = (np.cumsum(np.insert(count,0,0)[:-1])
                + np.random.randint(0, count.max(), count.size) % count)
    idx_uniq = idx_sort[idx_sel]
    orig2vox = np.zeros(len(coord), dtype=np.int64)
    orig2vox[idx_sort] = inverse
    return coord[idx_uniq], color[idx_uniq], grid[idx_uniq], orig2vox


@torch.no_grad()
def run_model(model, coord, color, grid, device):
    """Run model on voxelized tile. Returns per-voxel predictions."""
    feat = np.concatenate([coord, color], axis=1).astype(np.float32)
    out  = model(dict(
        coord      = torch.from_numpy(coord).float().to(device),
        feat       = torch.from_numpy(feat).float().to(device),
        grid_coord = torch.from_numpy(grid).int().to(device),
        offset     = torch.tensor([len(coord)], dtype=torch.int32).to(device),
    ))
    logits = out["seg_logits"] if isinstance(out, dict) else out
    return torch.argmax(logits, dim=1).cpu().numpy()


@torch.no_grad()
def predict_tile_v2(model_v1, model_v2, coord, color, grid_size, device):
    """
    Two-stage prediction:
    1. Run v1 on tile -> get v1_pred per original point
    2. Add v1_pred as extra feature
    3. Run v2 on tile with [intensity, v1_pred] -> get final prediction
    """
    v_coord, v_color, v_grid, orig2vox = voxelize(coord, color, grid_size)

    # Stage 1: v1 prediction
    pred_v1_vox = run_model(model_v1, v_coord, v_color, v_grid, device)

    # Stage 2: add v1_pred as extra feature, normalized to [0,1]
    v1_norm   = (pred_v1_vox / 9.0).reshape(-1, 1).astype(np.float32)
    color_v2  = np.concatenate([v_color, v1_norm], axis=1)  # (N, 2)

    # Stage 3: v2 prediction
    pred_v2_vox = run_model(model_v2, v_coord, color_v2, v_grid, device)

    return pred_v2_vox[orig2vox]


def predict_file(model_v1, model_v2, file_path, output_dir,
                 tile_size, tile_threshold, grid_size, device):
    stem     = Path(file_path).stem
    out_path = os.path.join(output_dir, f"Refine45PRED_{stem}.laz")

    if os.path.exists(out_path):
        print(f"  [SKIP] {out_path}")
        return

    try:
        las = laspy.read(file_path)
    except Exception as e:
        print(f"  [ERROR] {file_path}: {e}")
        return

    x = np.array(las.x, dtype=np.float64)
    y = np.array(las.y, dtype=np.float64)
    z = np.array(las.z, dtype=np.float64)
    n_total   = len(x)
    intensity = (np.array(las.intensity, dtype=np.float32) / 65535.0
                 if hasattr(las, "intensity")
                 else np.zeros(n_total, dtype=np.float32))

    global_mean    = np.array([x.mean(), y.mean(), z.mean()], dtype=np.float32)
    x_range        = x.max() - x.min()
    y_range        = y.max() - y.min()
    needs_chunking = (x_range > tile_threshold) or (y_range > tile_threshold)

    if needs_chunking:
        x_starts = np.arange(x.min(), x.max(), tile_size)
        y_starts = np.arange(y.min(), y.max(), tile_size)
        print(f"  {stem}: {n_total:,} pts -> {len(x_starts)*len(y_starts)} tiles")
    else:
        x_starts = [x.min()]
        y_starts = [y.min()]
        tile_size_x = x_range + 1.0
        tile_size_y = y_range + 1.0
        print(f"  {stem}: {n_total:,} pts -> already tiled")

    pred_all = np.zeros(n_total, dtype=np.int64)
    pbar     = tqdm(total=len(x_starts)*len(y_starts), desc="  Tiles", leave=False)

    for x0 in x_starts:
        for y0 in y_starts:
            x1   = x0 + (tile_size if needs_chunking else tile_size_x)
            y1   = y0 + (tile_size if needs_chunking else tile_size_y)
            mask = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
            pbar.update(1)
            if mask.sum() < 100:
                continue

            coord  = np.stack([x[mask].astype(np.float32),
                               y[mask].astype(np.float32),
                               z[mask].astype(np.float32)], axis=1)
            coord -= global_mean
            color  = intensity[mask].reshape(-1, 1)

            try:
                pred_tile      = predict_tile_v2(model_v1, model_v2, coord, color, grid_size, device)
                pred_all[mask] = pred_tile
            except RuntimeError as e:
                print(f"    [WARN] {e}")
                torch.cuda.empty_cache()

    pbar.close()

    pred_original = np.zeros_like(pred_all)
    for train_id, orig_id in TRAIN_TO_ORIGINAL.items():
        pred_original[pred_all == train_id] = orig_id

    header         = laspy.LasHeader(point_format=las.header.point_format, version=las.header.version)
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

    unique, counts = np.unique(pred_original, return_counts=True)
    print(f"  Saved -> {out_path}")
    print(f"  Class distribution:")
    for u, c in zip(unique, counts):
        print(f"    {u:2d}  {CLASS_NAMES.get(int(u),str(u)):15s}: {c:>10,}  ({c/n_total*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir",      required=True)
    parser.add_argument("--output_dir",     required=True)
    parser.add_argument("--weight_v1",      required=True, help="Path to v1 model weights")
    parser.add_argument("--weight_v2",      required=True, help="Path to v2 model weights")
    parser.add_argument("--tile_size",      type=float, default=25.0)
    parser.add_argument("--tile_threshold", type=float, default=30.0)
    parser.add_argument("--grid_size",      type=float, default=0.05)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_v1 = build_model_v1(args.weight_v1, device)
    model_v2 = build_model_v2(args.weight_v2, device)

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
        predict_file(model_v1, model_v2, fp, args.output_dir,
                     args.tile_size, args.tile_threshold,
                     args.grid_size, device)
    print("\nAll done.")

if __name__ == "__main__":
    main()
