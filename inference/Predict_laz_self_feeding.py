"""
predict_laz_iterative.py - PTv3 V2 self-feeding iterative inference with TTA + Post-Processing.

Added Features:
    --rescue_bollards  : Dynamically extracts AI bollard blueprints and rescues missed ones.
    --apply_csf        : Runs physics-based Cloth Simulation Filter to smooth ground bleed.
"""

import os
import argparse
import numpy as np
import torch
import laspy
from pathlib import Path
from tqdm import tqdm
from pointcept.models import build_model
from sklearn.cluster import DBSCAN  # Added for Bollard Rescue
import gc
try:
    import CSF                      # Added for Ground Smoothing
except ImportError:
    CSF = None

import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.cm as cm
try:
    import open3d as o3d
except ImportError:
    o3d = None
# CLASS DEFINITIONS (must match your training config)

TRAIN_TO_ORIGINAL = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:9, 9:10}
CLASS_NAMES = {
    0:  "unclassified",
    1:  "ground",
    2:  "pole",
    3:  "sign",
    4:  "bollard",
    5:  "trunk",
    6:  "vegetation",
    7:  "building",
    9:  "fence",
    10: "gate",
}
NUM_CLASSES  = 10
V2_MAX_CLASS = 9.0   # for normalizing predictions to [0, 1]



# TTA AUGMENTATIONS

def get_tta_augmentations(mode="normal"):
    identity = dict(rotate_z=0.0, flip_x=False, flip_y=False)
    if mode == "none": return [identity]
    elif mode == "fast":
        return [identity, dict(rotate_z=90.0, flip_x=False, flip_y=False),
                dict(rotate_z=180.0, flip_x=False, flip_y=False),
                dict(rotate_z=270.0, flip_x=False, flip_y=False)]
    elif mode == "normal":
        return [identity, dict(rotate_z=90.0, flip_x=False, flip_y=False),
                dict(rotate_z=180.0, flip_x=False, flip_y=False),
                dict(rotate_z=270.0, flip_x=False, flip_y=False),
                dict(rotate_z=0.0, flip_x=True, flip_y=False),
                dict(rotate_z=90.0, flip_x=True, flip_y=False),
                dict(rotate_z=0.0, flip_x=False, flip_y=True),
                dict(rotate_z=90.0, flip_x=False, flip_y=True)]
    elif mode == "full":
        return [identity, dict(rotate_z=45.0, flip_x=False, flip_y=False),
                dict(rotate_z=90.0, flip_x=False, flip_y=False),
                dict(rotate_z=135.0, flip_x=False, flip_y=False),
                dict(rotate_z=180.0, flip_x=False, flip_y=False),
                dict(rotate_z=225.0, flip_x=False, flip_y=False),
                dict(rotate_z=270.0, flip_x=False, flip_y=False),
                dict(rotate_z=315.0, flip_x=False, flip_y=False),
                dict(rotate_z=0.0, flip_x=True, flip_y=False),
                dict(rotate_z=90.0, flip_x=True, flip_y=False),
                dict(rotate_z=180.0, flip_x=True, flip_y=False),
                dict(rotate_z=270.0, flip_x=True, flip_y=False),
                dict(rotate_z=0.0, flip_x=False, flip_y=True),
                dict(rotate_z=90.0, flip_x=False, flip_y=True),
                dict(rotate_z=0.0, flip_x=True, flip_y=True),
                dict(rotate_z=90.0, flip_x=True, flip_y=True)]
    else:
        raise ValueError(f"Unknown tta_mode '{mode}'")

def apply_augmentation(coord, aug):
    c = coord.copy()
    if aug["flip_x"]: c[:, 0] *= -1.0
    if aug["flip_y"]: c[:, 1] *= -1.0
    angle_deg = aug["rotate_z"]
    if angle_deg != 0.0:
        angle_rad = np.deg2rad(angle_deg)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        x_new = cos_a * c[:, 0] - sin_a * c[:, 1]
        y_new = sin_a * c[:, 0] + cos_a * c[:, 1]
        c[:, 0], c[:, 1] = x_new, y_new
    return c


# MODEL & VOXELIZATION

def build_v2_model(weight_path, device):
    model_cfg = dict(
        type="DefaultSegmentorV2", num_classes=10, backbone_out_channels=64,
        backbone=dict(
            type="PT-v3m1", in_channels=5, order=("z", "z-trans", "hilbert", "hilbert-trans"),
            stride=(2, 2, 2, 2), enc_depths=(2, 2, 2, 6, 2), enc_channels=(32, 64, 128, 256, 512),
            enc_num_head=(2, 4, 8, 16, 32), enc_patch_size=(48, 48, 48, 48, 48),
            dec_depths=(2, 2, 2, 2), dec_channels=(64, 64, 128, 256),
            dec_num_head=(4, 4, 8, 16), dec_patch_size=(48, 48, 48, 48),
            mlp_ratio=4, qkv_bias=True, qk_scale=None, attn_drop=0.0, proj_drop=0.0, drop_path=0.3,
            shuffle_orders=True, pre_norm=True, enable_rpe=False, enable_flash=False,
            upcast_attention=False, upcast_softmax=False,
        ),
        criteria=[dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=255)],
    )
    model = build_model(model_cfg)
    ckpt  = torch.load(weight_path, map_location=device, weights_only=False)
    sd    = ckpt.get("state_dict", ckpt.get("model", ckpt))
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    print(f"  V2 model loaded: {weight_path}")
    return model

def voxelize(coord, feat, grid_size=0.05):
    scaled = coord / grid_size
    grid = np.floor(scaled).astype(np.int64)
    grid -= grid.min(axis=0)
    key = (grid[:, 0].astype(np.int64) << 40) | (grid[:, 1].astype(np.int64) << 20) | grid[:, 2].astype(np.int64)
    idx_sort = np.argsort(key)
    _, inverse, count = np.unique(key[idx_sort], return_inverse=True, return_counts=True)
    idx_sel = np.cumsum(np.insert(count, 0, 0)[:-1]) + np.random.randint(0, count.max(), count.size) % count
    idx_uniq = idx_sort[idx_sel]
    orig2vox = np.zeros(len(coord), dtype=np.int64)
    orig2vox[idx_sort] = inverse
    return coord[idx_uniq], feat[idx_uniq], grid[idx_uniq], orig2vox

@torch.no_grad()
def forward_pass(model, coord_aug, feat, grid_size, device):
    v_coord, v_feat, v_grid, orig2vox = voxelize(coord_aug, feat, grid_size)
    full_feat = np.concatenate([v_coord, v_feat], axis=1).astype(np.float32)
    input_dict = dict(
        coord=torch.from_numpy(v_coord).float().to(device),
        feat=torch.from_numpy(full_feat).float().to(device),
        grid_coord=torch.from_numpy(v_grid).int().to(device),
        offset=torch.tensor([len(v_coord)], dtype=torch.int32).to(device),
    )
    out = model(input_dict)
    logits = out["seg_logits"] if isinstance(out, dict) else out
    probs = torch.softmax(logits, dim=1).cpu().numpy()
    torch.cuda.empty_cache()
    return probs[orig2vox]


def predict_tile_iterative(model, coord, intensity, grid_size, device, n_iterations, augmentations):
    n_points = len(coord)
    v1_pred_norm = np.zeros((n_points, 1), dtype=np.float32)

    for iteration in range(1, n_iterations + 1):
        feat = np.concatenate([intensity, v1_pred_norm], axis=1)
        is_final = (iteration == n_iterations)

        if is_final:
            prob_sum = np.zeros((n_points, NUM_CLASSES), dtype=np.float32)
            for aug in augmentations:
                coord_aug = apply_augmentation(coord, aug)
                prob_sum += forward_pass(model, coord_aug, feat, grid_size, device)
            pred = np.argmax(prob_sum / len(augmentations), axis=1)
        else:
            probs = forward_pass(model, coord, feat, grid_size, device)
            pred  = np.argmax(probs, axis=1)
            v1_pred_norm = (pred.astype(np.float32) / V2_MAX_CLASS).reshape(-1, 1)

    return pred


# POST-PROCESSING INJECTIONS

def rescue_phantom_bollards(coord, pred):
    """ACTUAL RESCUE ALGORITHM (Production Ready - Full Upgrades)"""
    UNCLASS_ID = 0
    GROUND_ID = 1  # AI class label for Ground
    BOLLARD_ID = 4 # AI class label for Bollard
    
    print("\n  [DEBUG] Step 1: Finding 'Golden Reference' from AI-predicted bollards...")
    
    bollard_mask = (pred == BOLLARD_ID)
    bollard_coords = coord[bollard_mask]
    
    # UPGRADE 1: The "Ghost Town" Early Exit
    if len(bollard_coords) == 0:
        print("    [!] AI predicted 0 bollards. Assuming empty street. Aborting rescue.")
        return pred
        
    # Fallback thresholds in case AI bollards are corrupted
    ref_h, ref_w, ref_c = 1.0, 0.25, 0.20 
    
    # Separate the known bollards to measure them
    db_known = DBSCAN(eps=0.05, min_samples=20).fit(bollard_coords[:, :2])
    h_list, w_list, c_list = [], [], []
    
    for b_id in set(db_known.labels_):
        if b_id == -1: continue
        b_pts = bollard_coords[db_known.labels_ == b_id]
        
        h = b_pts[:, 2].max() - b_pts[:, 2].min()
        w = max(b_pts[:, 0].max() - b_pts[:, 0].min(), b_pts[:, 1].max() - b_pts[:, 1].min())
        xy_cen = b_pts[:, :2] - np.mean(b_pts[:, :2], axis=0)
        _, sv, _ = np.linalg.svd(xy_cen, full_matrices=False)
        c = sv[1] / (sv[0] + 1e-6) if len(sv) >= 2 else 0.0
        
        # UPGRADE 2: The Sanity Check
        if (h > 0.35) and (0.10 < w < 0.50) and (c > 0.40):
            h_list.append(h)
            w_list.append(w)
            c_list.append(c)
        
    if h_list:
        # 90th percentile = "Reasonable Highest"
        ref_h = np.percentile(h_list, 90)
        ref_w = np.percentile(w_list, 90)
        ref_c = np.percentile(c_list, 90)
        print(f"    ✓ Found {len(h_list)} VALID distinct bollards in the AI prediction.")
        print(f"    ✓ REFERENCE VALUES -> Height: {ref_h:.2f}m | Width: {ref_w:.2f}m | Curve: {ref_c:.2f}")
    else:
        # UPGRADE 3: The Safe Override
        print("    [!] AI bollards failed sanity check (garbage dimensions). Using fallback values.")
        print(f"    ✓ FALLBACK VALUES -> Height: {ref_h:.2f}m | Width: {ref_w:.2f}m | Curve: {ref_c:.2f}")

    # STEP 2: APPLY THE 20% BUFFER 
    BUFFER = 0.20
    min_h, max_h = ref_h * (1.0 - BUFFER), ref_h * (1.0 + BUFFER)
    min_w, max_w = ref_w * (1.0 - BUFFER), ref_w * (1.0 + BUFFER)
    min_c = ref_c * (1.0 - BUFFER) 

    print(f"\n  [DEBUG] Step 2: Scanning Unclassified Points with Dynamic Ceilings...")
    
    unclass_mask = (pred == UNCLASS_ID)
    unclass_coords = coord[unclass_mask]
    unclass_indices = np.where(unclass_mask)[0] 
    
    # We need ground points to build the ceilings
    ground_mask = (pred == GROUND_ID)
    ground_coords = coord[ground_mask]
    
    if len(unclass_coords) == 0:
        return pred

    chunk_size = 10.0
    sub_grid_size = 2.0 # 2x2 meter mini-grids for handling stairs/hills
    
    x_min, y_min = unclass_coords[:, 0].min(), unclass_coords[:, 1].min()
    x_max, y_max = unclass_coords[:, 0].max(), unclass_coords[:, 1].max()
    x_bins = np.arange(x_min, x_max + chunk_size, chunk_size)
    y_bins = np.arange(y_min, y_max + chunk_size, chunk_size)

    chunk_id = 0
    total_rescued = 0

    for i in range(len(x_bins) - 1):
        for j in range(len(y_bins) - 1):
            x0, x1 = x_bins[i], x_bins[i+1]
            y0, y1 = y_bins[j], y_bins[j+1]

            chunk_u_mask = (unclass_coords[:, 0] >= x0) & (unclass_coords[:, 0] < x1) & \
                           (unclass_coords[:, 1] >= y0) & (unclass_coords[:, 1] < y1)

            if chunk_u_mask.sum() < 10:
                continue

            chunk_g_mask = (ground_coords[:, 0] >= x0) & (ground_coords[:, 0] < x1) & \
                           (ground_coords[:, 1] >= y0) & (ground_coords[:, 1] < y1)

            chunk_u_pts = unclass_coords[chunk_u_mask]
            chunk_u_idx = unclass_indices[chunk_u_mask]
            chunk_g_pts = ground_coords[chunk_g_mask]
            
            #  UPGRADE 4: The Levitation Rule (Chunk Level) 
            if len(chunk_g_pts) == 0:
                continue
                
            chunk_median_floor = np.median(chunk_g_pts[:, 2])
            
            # UPGRADE 5: 2x2 Sub-Grid Dynamic Ceilings
            valid_u_pts_list = []
            valid_u_idx_list = []
            
            for sx in np.arange(x0, x1, sub_grid_size):
                for sy in np.arange(y0, y1, sub_grid_size):
                    sg_u_mask = (chunk_u_pts[:, 0] >= sx) & (chunk_u_pts[:, 0] < sx + sub_grid_size) & \
                                (chunk_u_pts[:, 1] >= sy) & (chunk_u_pts[:, 1] < sy + sub_grid_size)
                    
                    if not sg_u_mask.any():
                        continue
                        
                    sg_g_mask = (chunk_g_pts[:, 0] >= sx) & (chunk_g_pts[:, 0] < sx + sub_grid_size) & \
                                (chunk_g_pts[:, 1] >= sy) & (chunk_g_pts[:, 1] < sy + sub_grid_size)
                    
                    if sg_g_mask.any():
                        local_floor = np.median(chunk_g_pts[sg_g_mask, 2])
                    else:
                        local_floor = chunk_median_floor
                        
                    z_ceiling = local_floor + max_h + 0.10
                    
                    sg_u_pts = chunk_u_pts[sg_u_mask]
                    sg_u_idx = chunk_u_idx[sg_u_mask]
                    
                    below_ceiling_mask = (sg_u_pts[:, 2] <= z_ceiling)
                    valid_u_pts_list.append(sg_u_pts[below_ceiling_mask])
                    valid_u_idx_list.append(sg_u_idx[below_ceiling_mask])

            if not valid_u_pts_list:
                continue
                
            filtered_chunk_pts = np.vstack(valid_u_pts_list)
            filtered_chunk_idx = np.concatenate(valid_u_idx_list)
            
            if len(filtered_chunk_pts) < 10:
                continue

            chunk_id += 1
            
            xy_chunk_pts = filtered_chunk_pts[:, :2]
            db_un = DBSCAN(eps=0.08, min_samples=50) 
            un_labels = db_un.fit_predict(xy_chunk_pts)

            unique_clusters = set(un_labels)
            unique_clusters.discard(-1)

            for c_id in unique_clusters:
                cluster_local_mask = (un_labels == c_id)
                c_pts = filtered_chunk_pts[cluster_local_mask]
                cluster_global_idx = filtered_chunk_idx[cluster_local_mask] 
                
                h = c_pts[:, 2].max() - c_pts[:, 2].min()
                w = max(c_pts[:, 0].max() - c_pts[:, 0].min(), c_pts[:, 1].max() - c_pts[:, 1].min())
                xy_cluster = c_pts[:, :2]
                xy_centered = xy_cluster - np.mean(xy_cluster, axis=0)
                _, sv, _ = np.linalg.svd(xy_centered, full_matrices=False)
                curve = sv[1] / (sv[0] + 1e-6) if len(sv) >= 2 else 0.0
                
                # NEW: The "Rooted to the Ground" Check 
                c_min_z = c_pts[:, 2].min()
                cx, cy = np.mean(c_pts[:, 0]), np.mean(c_pts[:, 1])
                
                # Look for ground directly under this specific object (within 1 meter)
                local_g_mask = (np.abs(chunk_g_pts[:, 0] - cx) < 1.0) & (np.abs(chunk_g_pts[:, 1] - cy) < 1.0)
                cluster_floor = np.median(chunk_g_pts[local_g_mask, 2]) if local_g_mask.any() else chunk_median_floor
                
                # Ensure the bottom of the object is within 10cm of the floor
                is_rooted = (c_min_z <= cluster_floor + 0.10)
                
                # ACTUAL RESCUE LOGIC 
                # Added 'is_rooted' to the strict requirements
                if (min_h <= h <= max_h) and (min_w <= w <= max_w) and (curve >= min_c) and is_rooted:
                    total_rescued += 1
                    
                    # MODIFY THE PREDICTION ARRAY
                    pred[cluster_global_idx] = BOLLARD_ID
                    
                    # LOG IT TO TERMINAL
                    print(f"    -> [SUCCESS] Rescued Bollard in Chunk {chunk_id}! (H:{h:.2f}, W:{w:.2f}, C:{curve:.2f})")

    print(f"\n  [DEBUG] Process Complete! Successfully rescued {total_rescued} phantom bollards.")
    
    return pred

def apply_dbscan_smoothing(coord, pred):
    """Removes jagged 'salt and pepper' noise using DBSCAN and KNN."""
    print("\n  [Post-Process] Running DBSCAN edge smoothing...")
    
    # Classes we want to clean (1: Ground, 7: Building)
    # You can add others like Pole (2) if they are noisy!
    classes_to_clean = [1, 7] 
    
    smoothed_pred = pred.copy()
    total_fixed = 0
    
    for cls_id in classes_to_clean:
        mask = (smoothed_pred == cls_id)
        if not mask.any(): continue
        
        cls_coords = coord[mask]
        cls_indices = np.where(mask)[0]
        
        # 1. Run DBSCAN to find isolated/stray points
        # eps=0.15 (15cm), min_samples=10. 
        # If a point doesn't have 10 friends within 15cm, it's flagged as noise (-1)!
        db = DBSCAN(eps=0.15, min_samples=10, n_jobs=-1)
        labels = db.fit_predict(cls_coords[:, :3])
        
        noise_local_mask = (labels == -1)
        noise_global_indices = cls_indices[noise_local_mask]
        
        if len(noise_global_indices) == 0:
            continue
            
        # 2. Reassign noise points using their nearest "stable" neighbors
        # We build a KDTree out of all the points that are NOT noise
        stable_mask = ~np.isin(np.arange(len(pred)), noise_global_indices)
        stable_coords = coord[stable_mask]
        stable_preds = smoothed_pred[stable_mask]
        
        # Fast spatial query
        tree = cKDTree(stable_coords)
        noise_coords = coord[noise_global_indices]
        
        # Find the single closest stable point for each noise point
        _, nearest_idx = tree.query(noise_coords, k=1)
        
        # Overwrite the jagged noise point with the class of its nearest stable neighbor
        smoothed_pred[noise_global_indices] = stable_preds[nearest_idx]
        
        total_fixed += len(noise_global_indices)
        print(f"    ✓ Cleaned {len(noise_global_indices):,} jagged points for class {cls_id}.")
        
    print(f"    ✓ Total edges smoothed: {total_fixed:,} points.")
    return smoothed_pred

def apply_csf_ground_smoothing(coord, pred):
    """Runs a physics simulation to smooth ground bleed under cars and edges."""
    print("\n  [Post-Process] Running Cloth Simulation Filter (Ground Smoothing)...")
    if CSF is None:
        print("    [ERROR] CSF library not found. Run: pip install CSF")
        return pred

    # Drop the cloth
    csf = CSF.CSF()
    csf.params.bSloopSmooth = False
    csf.params.cloth_resolution = 0.5
    csf.params.rigidness = 2 #(1-3)lower number represents soft cloth. 1 represents Bengali moslin or silk.
    csf.params.class_threshold = 0.03 # Points within 10cm of the cloth are topological ground

    csf.setPointCloud(coord)
    ground_indices = CSF.VecInt()
    non_ground_indices = CSF.VecInt()
    csf.do_filtering(ground_indices, non_ground_indices, exportCloth=False)

    g_idx = np.array(ground_indices)

    # We overwrite the AI's prediction if the physics engine proves it's flat ground.
    # HOWEVER: We protect thin vertical objects (Poles, Signs, Bollards, Trunks) 
    # just in case the cloth accidentally touched their very base.
    protected_classes = [2, 3, 4, 5] 
    
    mask = np.isin(pred[g_idx], protected_classes, invert=True)
    valid_g_idx = g_idx[mask]

    # Calculate how many points we fixed (for the print log)
    GROUND_ID = 1
    changed_points = (pred[valid_g_idx] != GROUND_ID).sum()
    
    # Overwrite
    pred[valid_g_idx] = GROUND_ID

    print(f"    ✓ CSF smoothed {changed_points:,} bleed points back into 'ground'.")
    return pred


# PROCESS ONE FILE
def predict_file(model, file_path, output_dir, args, augmentations, device):
    stem = Path(file_path).stem
    out_path = os.path.join(output_dir, f"ITER_{stem}.laz")

    if os.path.exists(out_path):
        print(f"  [SKIP] Already exists: {out_path}")
        return

    try:
        las = laspy.read(file_path)
    except Exception as e:
        print(f"  [ERROR] Cannot read: {e}")
        return

    x, y, z = np.array(las.x, dtype=np.float64), np.array(las.y, dtype=np.float64), np.array(las.z, dtype=np.float64)
    n_total = len(x)
    global_z_ground = np.percentile(z, 1)

    intensity = (np.array(las.intensity, dtype=np.float32) / 65535.0 if hasattr(las, "intensity") else np.zeros(n_total, dtype=np.float32)).reshape(-1, 1)

    x_range, y_range = x.max() - x.min(), y.max() - y.min()
    needs_chunking = (x_range > args.tile_threshold) or (y_range > args.tile_threshold)

    x_starts = np.arange(x.min(), x.max(), args.tile_size) if needs_chunking else [x.min()]
    y_starts = np.arange(y.min(), y.max(), args.tile_size) if needs_chunking else [y.min()]
    tile_size_x, tile_size_y = (args.tile_size, args.tile_size) if needs_chunking else (x_range + 1.0, y_range + 1.0)

    pred_all = np.zeros(n_total, dtype=np.int64)

    pbar = tqdm(total=len(x_starts) * len(y_starts), desc=f"  Tiles ({stem})", leave=False)

    for x0 in x_starts:
        for y0 in y_starts:
            x1, y1 = x0 + tile_size_x, y0 + tile_size_y
            margin = 2.0 
            
            inner_mask = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
            outer_mask = (x >= x0 - margin) & (x < x1 + margin) & (y >= y0 - margin) & (y < y1 + margin)

            pbar.update(1)
            if inner_mask.sum() < 100: continue

            tile_x, tile_y, tile_z = x[outer_mask].astype(np.float32), y[outer_mask].astype(np.float32), z[outer_mask].astype(np.float32)
            coord = np.stack([tile_x, tile_y, tile_z - global_z_ground], axis=1)
            coord[:, 0] -= tile_x.mean()
            coord[:, 1] -= tile_y.mean()

            try:
                pred_tile_outer = predict_tile_iterative(model, coord, intensity[outer_mask], args.grid_size, device, args.iterations, augmentations)
                is_inner = inner_mask[outer_mask]
                pred_all[inner_mask] = pred_tile_outer[is_inner]
            except RuntimeError as e:
                if "out of memory" in str(e).lower(): print(f"\n  [OOM] Try --tile_size 15")
                torch.cuda.empty_cache()
    pbar.close()

    # Remap train IDs to original LAZ class IDs 
    pred_original = np.zeros_like(pred_all)
    for train_id, orig_id in TRAIN_TO_ORIGINAL.items():
        pred_original[pred_all == train_id] = orig_id

    # NEW: INJECTION POINT FOR POST-PROCESSING
    if args.rescue_bollards or args.apply_csf:
        # We need the global coordinates to do physical measurements
        global_coord = np.stack([x, y, z], axis=1)

        if args.apply_csf:
            pred_original = apply_csf_ground_smoothing(global_coord, pred_original)
         
        if args.rescue_bollards:
            pred_original = rescue_phantom_bollards(global_coord, pred_original)
            
        if args.rescue_bollards:
            pred_original = rescue_phantom_bollards(global_coord, pred_original)

    #Class distribution 
    print(f"\n  Class distribution:")
    unique, counts = np.unique(pred_original, return_counts=True)
    for u, c in zip(unique, counts):
        name = CLASS_NAMES.get(int(u), f"Unknown({u})")
        print(f"  {u:<5} {name:<15} {c:<12,} {c/n_total*100:.1f}%")

    # Save LAZ
    header = laspy.LasHeader(point_format=las.header.point_format, version=las.header.version)
    header.scales, header.offsets = las.header.scales, las.header.offsets
    out_las = laspy.LasData(header=header)

    for dim in las.point_format.dimension_names:
        try: setattr(out_las, dim, getattr(las, dim))
        except Exception: pass

    out_las.classification = pred_original.astype(np.uint8)
    out_las.write(out_path)
    print(f"\n  ✓ Saved → {out_path}")


# MAIN
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--weight", required=True)
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input_dir", type=str)
    group.add_argument("--single_file", type=str)

    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--tta_mode", type=str, default="normal", choices=["none", "fast", "normal", "full"])
    parser.add_argument("--tile_size", type=float, default=25.0)
    parser.add_argument("--tile_threshold", type=float, default=30.0)
    parser.add_argument("--grid_size", type=float, default=0.05)
    
    # NEW ARGUMENTS
    parser.add_argument("--rescue_bollards", action="store_true", help="Extract bollard blueprint to rescue unclassified ones")
    parser.add_argument("--apply_csf", action="store_true", help="Run CSF physics to smooth ground edges")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_v2_model(args.weight, device)
    augmentations = get_tta_augmentations(args.tta_mode)

    if args.single_file:
        predict_file(model, args.single_file, args.output_dir, args, augmentations, device)
    else:
        files = sorted([os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) if f.lower().endswith((".laz", ".las"))])
        for fp in files:
            predict_file(model, fp, args.output_dir, args, augmentations, device)

if __name__ == "__main__":
    main()