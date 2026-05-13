"""
pointcept_classifier.py - PSANP classifier for Pointcept PTv3 semantic segmentation.

Reads model architecture directly from your training config.py — no hardcoding.
Imports inference functions from Predict_laz_self_feeding.py — no duplication.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Set

import numpy as np
import torch

from geosat.cloud import PointCloud
from geosat.psanp.abstractions.classifier import Classifier
from geosat.psanp.abstractions.option import Option
from geosat.psanp.abstractions.pipelinemetadata import PipelineMetadata
from geosat.psanp.abstractions.stages import register_stage

logger = logging.getLogger(__name__)


# ==============================================================================
# HELPERS
# ==============================================================================

def _load_pointcept_config(config_path: Path):
    try:
        from pointcept.utils.config import Config
        return Config.fromfile(str(config_path))
    except Exception as e:
        raise RuntimeError(f"Failed to load config from {config_path}. Error: {e}")

def _get_grid_size_from_config(cfg) -> float:
    try:
        for t in cfg.data.train.transform:
            if t.get("type") == "GridSample":
                return float(t["grid_size"])
    except Exception:
        pass
    logger.warning("Could not read grid_size from config. Using default 0.05.")
    return 0.05

def _get_class_names_from_config(cfg) -> dict:
    return {i: name.lower() for i, name in enumerate(cfg.data.names)}

def _build_model_from_config(cfg, weight_path: Path, device: torch.device):
    try:
        from pointcept.models import build_model
    except ImportError as e:
        raise ImportError("Pointcept is not installed or not on PYTHONPATH.") from e

    model = build_model(cfg.model)
    ckpt  = torch.load(weight_path, map_location=device, weights_only=False)
    sd    = ckpt.get("state_dict", ckpt.get("model", ckpt))
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model loaded: %s (%.1fM params)", weight_path, n_params / 1e6)
    return model

def _import_inference_functions():
    """Import inference functions AND the new post-processing functions."""
    try:
        # Make sure your inference file is named correctly here based on your workspace
        from inference.Predict_laz_self_feeding import (
            get_tta_augmentations,
            apply_augmentation,
            voxelize,
            predict_tile_iterative,
            rescue_phantom_bollards,      # <-- NEW
            apply_csf_ground_smoothing    # <-- NEW
        )
        return (get_tta_augmentations, apply_augmentation, voxelize, 
                predict_tile_iterative, rescue_phantom_bollards, apply_csf_ground_smoothing)
    except ImportError as e:
        raise ImportError(f"Could not import from inference script. Error: {e}")


# ==============================================================================
# CLASSIFIER
# ==============================================================================

class PointceptClassifier(Classifier):

    def __init__(
        self,
        stage_name: Optional[str] = None,
        tta_mode: str = "normal",
        iterations: int = 3,
        tile_size: float = 25.0,
        tile_threshold: float = 30.0,
        rescue_bollards: bool = False,  # <-- NEW
        apply_csf: bool = False,        # <-- NEW
    ) -> None:
        super().__init__(stage_name)

        self.tta_mode        = tta_mode
        self.iterations      = iterations
        self.tile_size       = tile_size
        self.tile_threshold  = tile_threshold
        self.rescue_bollards = rescue_bollards
        self.apply_csf       = apply_csf

        bundle_folder = os.environ.get("PSANP_BUNDLE_FOLDER", "")
        if not bundle_folder:
            raise EnvironmentError("PSANP_BUNDLE_FOLDER is not set.")
        if stage_name is None:
            raise ValueError("'stage_name' is None")
            
        bundle_path = Path(bundle_folder) / stage_name
        weight_path = bundle_path / "model" / "model_best.pth"
        config_path = bundle_path / "config.py"

        cfg              = _load_pointcept_config(config_path)
        self.grid_size   = _get_grid_size_from_config(cfg)
        self.class_names = _get_class_names_from_config(cfg)

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = _build_model_from_config(cfg, weight_path, self.device)

        # ── Import inference functions ────────────────────────────
        (
            get_tta_augmentations,
            self._apply_augmentation,
            self._voxelize,
            self._predict_tile_iterative,
            self._rescue_phantom_bollards,      # <-- NEW
            self._apply_csf_ground_smoothing    # <-- NEW
        ) = _import_inference_functions()

        self.augmentations = get_tta_augmentations(tta_mode)

    @classmethod
    def options(cls) -> List[Option]:
        return [
            Option(
                "tta_mode",description="TTA mode: none / fast / normal / full. Default: normal",type_cast=str,required=False,
            ),
            Option(
                "iterations",description="Self-feeding passes before final TTA pass. Default: 3",type_cast=int,required=False,
            ),
            Option(
                "tile_size",description="Chunk size in metres for large point clouds. Default: 25.0",type_cast=float,required=False,
            ),
            Option(
                "tile_threshold",description="Point clouds larger than this get chunked. Default: 30.0",type_cast=float,required=False,
            ),
            Option(
                "rescue_bollards", description="Run DBSCAN to rescue missed bollards", type_cast=bool,required=False,
            ),
            Option(
                "apply_csf", description="Run Cloth Simulation Filter to smooth ground", type_cast=bool, required=False,
            ),
        ]

    def input_classes(self) -> Set[str]:
        return set()

    def output_classes(self) -> Set[str]:
        return set(self.class_names.values())

    def execute_on(
        self,
        cloud: PointCloud,
        pipeline_metadata: PipelineMetadata,
    ) -> Iterable[PointCloud]:

        if not len(cloud):
            return (cloud,)

        x = cloud.coords[:, 0].astype(np.float64)
        y = cloud.coords[:, 1].astype(np.float64)
        z = cloud.coords[:, 2].astype(np.float64)
        n_total = len(x)

        intensity_raw = np.asarray(cloud.intensity, dtype=np.float32)
        intensity = (intensity_raw / 65535.0 if intensity_raw.max() > 1.0 else intensity_raw).reshape(-1, 1)

        codes = pipeline_metadata.classification_codes
        train_id_to_code = {
            train_id: codes.get(class_name, codes.get("unclassified", 0))
            for train_id, class_name in self.class_names.items()
        }

        x_range = x.max() - x.min()
        y_range = y.max() - y.min()
        needs_chunking = (x_range > self.tile_threshold or y_range > self.tile_threshold)

        if needs_chunking:
            x_starts = np.arange(x.min(), x.max(), self.tile_size)
            y_starts = np.arange(y.min(), y.max(), self.tile_size)
        else:
            x_starts, y_starts = [x.min()], [y.min()]
            tile_size_x, tile_size_y = x_range + 1.0, y_range + 1.0

        pred_all = np.full(n_total, codes.get("unclassified", 0), dtype=np.int64)

        for x0 in x_starts:
            for y0 in y_starts:
                x1 = x0 + (self.tile_size if needs_chunking else tile_size_x)
                y1 = y0 + (self.tile_size if needs_chunking else tile_size_y)

                mask = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
                if mask.sum() < 100: continue

                tile_z = z[mask].astype(np.float32)
                z_ground_level = np.percentile(tile_z, 1)

                coord = np.stack([x[mask].astype(np.float32), y[mask].astype(np.float32), tile_z - z_ground_level], axis=1)
                coord[:, 0] -= coord[:, 0].mean()
                coord[:, 1] -= coord[:, 1].mean()

                try:
                    pred_tile = self._predict_tile_iterative(
                        self.model, coord, intensity[mask], self.grid_size, self.device, self.iterations, self.augmentations
                    )
                    pred_all[mask] = np.vectorize(train_id_to_code.get)(pred_tile)
                except RuntimeError as e:
                    logger.error("Tile failed: %s", e)
                    torch.cuda.empty_cache()

        # ==========================================================
        # POST-PROCESSING INJECTION
        # ==========================================================
        if self.apply_csf:
            logger.info("Applying CSF ground smoothing...")
            pred_all = self._apply_csf_ground_smoothing(cloud.coords, pred_all)
            
        if self.rescue_bollards:
            logger.info("Rescuing phantom bollards...")
            pred_all = self._rescue_phantom_bollards(cloud.coords, pred_all)

        # Write predictions back to PointCloud
        cloud.classification[:] = pred_all.astype(np.uint8)

        return (cloud,)

def _register_stages() -> None:
    register_stage(PointceptClassifier)

_register_stages()