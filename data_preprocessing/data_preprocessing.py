import laspy
import numpy as np
import os
import yaml
import argparse
from glob import glob
from tqdm import tqdm


def convert_laz_to_npy(file_path, output_dir, class_map):
    """
    Convert a single LAZ tile into a Pointcept-compatible scene folder.

    Output structure:
        output_dir/
            tile_name/
                coord.npy       (N, 3) float32  — xyz centered by mean
                color.npy       (N, 1) float32  — intensity normalised [0, 1]
                segment.npy     (N,)   int16     — remapped class labels
    """
    tile_name = os.path.basename(file_path).replace('.laz', '').replace('.las', '')
    scene_dir = os.path.join(output_dir, tile_name)

    # Skip if already processed
    if os.path.exists(os.path.join(scene_dir, 'segment.npy')):
        return

    try:
        las = laspy.read(file_path)
    except Exception as e:
        print(f"[ERROR] Could not read {file_path}: {e}")
        return

    # ---- Coordinates ----
    coords = np.vstack((las.x, las.y, las.z)).T.astype(np.float32)
    # Centre by mean (standard Pointcept convention)
    coords -= coords.mean(axis=0)

    # ---- Intensity as feature ----
    if hasattr(las, 'intensity'):
        intensity = np.array(las.intensity, dtype=np.float32)
        intensity /= 65535.0
    else:
        print(f"  [WARN] {tile_name} has no intensity field — using zeros.")
        intensity = np.zeros(coords.shape[0], dtype=np.float32)

    # Shape: (N, 1) — Pointcept expects (N, C) for color/feat
    color = intensity.reshape(-1, 1)

    # ---- Labels ----
    if not hasattr(las, 'classification'):
        print(f"  [SKIP] {tile_name} has no classification field.")
        return

    raw_labels = np.array(las.classification, dtype=np.int32)

    # Start with 255 = ignore for everything not in class_map
    new_labels = np.full(raw_labels.shape, 255, dtype=np.int16)
    for raw_id, train_id in class_map.items():
        mask = (raw_labels == int(raw_id))
        new_labels[mask] = int(train_id)

    # Sanity check
    n_ignored = (new_labels == 255).sum()
    n_total   = len(new_labels)
    if n_ignored / n_total > 0.5:
        print(f"  [WARN] {tile_name}: {n_ignored/n_total*100:.1f}% of points are ignored — "
              f"check your class_map.")

    # ---- Save ----
    os.makedirs(scene_dir, exist_ok=True)
    np.save(os.path.join(scene_dir, 'coord.npy'),   coords)
    np.save(os.path.join(scene_dir, 'color.npy'),   color)
    np.save(os.path.join(scene_dir, 'segment.npy'), new_labels)


def main():
    parser = argparse.ArgumentParser(
        description='Convert LAZ tiles to Pointcept numpy format'
    )
    parser.add_argument('--config', type=str, required=True,
                        help='Path to master YAML config')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        full_cfg = yaml.safe_load(f)

    if 'preprocessing' not in full_cfg:
        print("Error: config file missing 'preprocessing' section!")
        return

    cfg            = full_cfg['preprocessing']
    raw_root       = cfg['raw_root']
    processed_root = cfg['processed_root']
    class_map      = cfg['class_map']

    print(f"--- Preprocessing from config: {args.config} ---")
    print(f"    raw_root       : {raw_root}")
    print(f"    processed_root : {processed_root}")
    print(f"    class_map      : {class_map}")
    print()

    for split in ['train', 'val']:
        src = os.path.join(raw_root, split)
        dst = os.path.join(processed_root, split)

        if not os.path.exists(src):
            print(f"[WARNING] {src} not found, skipping.")
            continue

        os.makedirs(dst, exist_ok=True)
        files = sorted(glob(os.path.join(src, '*.laz')) +
                       glob(os.path.join(src, '*.las')))
        print(f"[{split.upper()}] Found {len(files)} files → {dst}")

        for f in tqdm(files, desc=split):
            convert_laz_to_npy(f, dst, class_map)

    print("\nDone. Verify one tile with:")
    print(f"  python -c \"import numpy as np; "
          f"d=np.load('{processed_root}/train/TILE_NAME/coord.npy'); print(d.shape)\"")


if __name__ == '__main__':
    main()