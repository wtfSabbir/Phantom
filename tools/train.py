"""
Main Training Script

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

from pointcept.engines.defaults import (
    default_argument_parser,
    default_config_parser,
    default_setup,
)
from pointcept.engines.train import TRAINERS
from pointcept.engines.launch import launch
from pointcept.mlflow_tools.mlflow_logger import setup_mlflow_logger
import glob
import os
import pointcept.utils.comm as comm


def main_worker(cfg):
    cfg = default_setup(cfg)

    # 1. Initialize logger as None so it's safe if MLflow is disabled
    logger = None
    
    # 2. Check BOTH that this is the main GPU AND that MLflow is enabled in config
    if comm.is_main_process() and cfg.get("use_mlflow", True):
        tracking_uri = "https://mlflow-dev.geo-sat.com"  
        experiment_name = "Pointcept"
        run_name = f"Refine_PTv3m1_Point180k"
            
        # Start the connection
        logger = setup_mlflow_logger(tracking_uri, experiment_name, run_name)
            
        # Extract the exact parameters from your lidar_config.py
        params_to_log = {
            "model_type": cfg.model.get("type", "None"),
            "backbone_type": cfg.model.backbone.get("type", "None"),
            "seed": cfg.get("seed", "None"),
            "num_worker": cfg.get("num_worker", "None"),
            "batch_size": cfg.get("batch_size", "None"),
            "batch_size_val": cfg.get("batch_size_val", "None"),
            "batch_size_test": cfg.get("batch_size_test", "None"),
            "eval_epoch": cfg.get("eval_epoch", "None"),
            "mix_prob": cfg.get("mix_prob", "None"),
            "clip_grad": cfg.get("clip_grad", "None"),
            "gradient_accumulation_steps": cfg.get("gradient_accumulation_steps", "None"),
            
            # Dataset Parameters
            "num_classes": cfg.data.get("num_classes", "None"),
            "class_names": str(cfg.data.get("names", "None")),
            "train_loop": cfg.data.train.get("loop", "None"),
            "val_loop": cfg.data.val.get("loop", "None"),
            "point_max": next((t.get("point_max") for t in cfg.data.train.transform if t.get("type") == "SphereCrop"), "None"),
            "grid_size": next((t.get("grid_size") for t in cfg.data.train.transform if t.get("type") == "GridSample"), "None"),
            
            # Backbone Dimensions
            "in_channels": cfg.model.backbone.get("in_channels", "None"),
            "backbone_out_channels": cfg.model.get("backbone_out_channels", "None"),
            "enc_depths": str(cfg.model.backbone.get("enc_depths", "None")),
            "enc_channels": str(cfg.model.backbone.get("enc_channels", "None")),
            "enc_num_head": str(cfg.model.backbone.get("enc_num_head", "None")),
            "enc_patch_size": str(cfg.model.backbone.get("enc_patch_size", "None")),
            "dec_depths": str(cfg.model.backbone.get("dec_depths", "None")),
            "dec_channels": str(cfg.model.backbone.get("dec_channels", "None")),
            "dec_num_head": str(cfg.model.backbone.get("dec_num_head", "None")),
            "dec_patch_size": str(cfg.model.backbone.get("dec_patch_size", "None")),
            
            # Attention Specs
            "mlp_ratio": cfg.model.backbone.get("mlp_ratio", "None"),
            "qkv_bias": cfg.model.backbone.get("qkv_bias", "None"),
            "qk_scale": cfg.model.backbone.get("qk_scale", "None"),
            "attn_drop": cfg.model.backbone.get("attn_drop", "None"),
            "proj_drop": cfg.model.backbone.get("proj_drop", "None"),
            "drop_path": cfg.model.backbone.get("drop_path", "None"),
            
            # Optimizer & Scheduler
            "optimizer_type": cfg.optimizer.get("type", "None"),
            "weight_decay": cfg.optimizer.get("weight_decay", "None"),
            "scheduler_type": cfg.scheduler.get("type", "None"),
            "max_lr": cfg.scheduler.get("max_lr", "None"),
            "pct_start": cfg.scheduler.get("pct_start", "None"),
            "div_factor": cfg.scheduler.get("div_factor", "None"),
            "final_div_factor": cfg.scheduler.get("final_div_factor", "None"),
            "total_steps": cfg.scheduler.get("total_steps", "None"),
        }
        logger.log_parameters(params_to_log)
        
        # Log your datasets
        logger.log_dataset(cfg.data.train.data_root, "PTv3_Train_Set", context="training")
        logger.log_dataset(cfg.data.val.data_root, "PTv3_Val_Set", context="evaluation")
        logger.log_dataset_details(cfg.data.train.data_root, prefix="Total")
        #logger.log_dataset_details(cfg.data.val.data_root, prefix="Val")
        train_folder = os.path.join(cfg.data.train.data_root, "train")
        val_folder = os.path.join(cfg.data.val.data_root, "val")
        
        logger.log_dataset_details(train_folder, prefix="Train")
        logger.log_dataset_details(val_folder, prefix="Val")

    # =======================================================
    # START TRAINING
    # =======================================================
    trainer = TRAINERS.build(dict(type=cfg.train.type, cfg=cfg))
    trainer.train()

    # 3. Check the condition again before finalizing the logs
    if comm.is_main_process() and cfg.get("use_mlflow", True) and logger is not None:
        # 1. Upload the Pointcept config backup 
        config_backup_path = os.path.join(cfg.save_path, "config.py")
        if os.path.exists(config_backup_path):
            logger.log_artifact(config_backup_path)
            print("[MLflow] Uploaded config.py backup.")

        # 2. Find and upload the train.log file (catches train.log, train.log.2026, etc.)
        log_files = glob.glob(os.path.join(cfg.save_path, "*.log"))
        if len(log_files) > 0:
            logger.log_artifact(log_files[0])
            print(f"[MLflow] Successfully uploaded log file: {log_files[0]}")
        else:
            print("[MLflow Warning] Could not find the train.log file to upload.")
            
        # Seal the envelope!
        logger.close()


def main():
    args = default_argument_parser().parse_args()
    cfg = default_config_parser(args.config_file, args.options)

    launch(
        main_worker,
        num_gpus_per_machine=args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        cfg=(cfg,),
    )


if __name__ == "__main__":
    main()