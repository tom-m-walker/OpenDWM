"""
Evaluation script for the LiDAR diffusion model.

Unlike preview.py (which calls preview_pipeline), this script calls
evaluate_pipeline, which:
  - Runs autoregressive inference when enable_autoregressive_inference=true
  - Saves predicted point clouds as .bin files when save_pred_results=true
  - Computes and prints metrics (chamfer distance, IoU, MMD, JSD)

Usage:
    PYTHONPATH=src python3 -m torch.distributed.run \
        --nnodes 1 --nproc-per-node 2 --node-rank 0 \
        --master-addr 127.0.0.1 --master-port 29000 \
        examples/lidar_diffusion_evaluate.py \
        -c examples/lidar_diffusion_nuscenes_val.json \
        -o output/lidar_nuscenes_eval
"""

import argparse
import dwm.common
import json
import os
import torch


def create_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate a LiDAR diffusion model and optionally save "
        "predicted point clouds.")
    parser.add_argument(
        "-c", "--config-path", type=str, required=True,
        help="Path to the config JSON file.")
    parser.add_argument(
        "-o", "--output-path", type=str, required=True,
        help="Directory to write outputs (point clouds, previews).")
    return parser


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    with open(args.config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    ddp = "LOCAL_RANK" in os.environ
    if ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(config["device"], local_rank)
        if config["device"] == "cuda":
            torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend=config["ddp_backend"])
    else:
        device = torch.device(config["device"])

    if "global_state" in config:
        for key, value in config["global_state"].items():
            dwm.common.global_state[key] = \
                dwm.common.create_instance_from_config(value)

    should_log = (ddp and local_rank == 0) or not ddp

    pipeline = dwm.common.create_instance_from_config(
        config["pipeline"], output_path=args.output_path, config=config,
        device=device)
    if should_log:
        print("The pipeline is loaded.")

    validation_dataset = dwm.common.create_instance_from_config(
        config["validation_dataset"])
    if should_log:
        print("The validation dataset is loaded with {} items.".format(
            len(validation_dataset)))

    dataloader_config = config.get(
        "validation_dataloader", config.get("preview_dataloader", {}))
    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        **dwm.common.instantiate_config(dataloader_config))

    os.makedirs(args.output_path, exist_ok=True)

    with torch.no_grad():
        pipeline.evaluate_pipeline(
            global_step=0,
            dataset_length=len(validation_dataset),
            validation_dataloader=validation_dataloader,
            log_type="tensorboard")

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
