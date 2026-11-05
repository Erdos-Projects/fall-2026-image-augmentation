# Copyright (C) 2026  The Ohio State University

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Train YOLO26n-pose on a hinge dataset (YOLO pose format; 3 keypoints: tip A, vertex, tip B).

Training protocol (the same for every model in this repo):
- starts from the COCO-pose YOLO26n weights (Weights/yolo26n-pose.pt; Ultralytics downloads them if missing);
- the validation Angle MAE is the fitness (angle_fitness.py): it picks best.pt and stops training 50 epochs after
  the best epoch (at most 500 epochs); validation runs on the square 224 px canvas, as prediction does;
- built-in augmentation is off; --degrees 180 turns on Ultralytics' random whole-image rotation.

Usage, from the repo root (datasets as described in guidance.md):
    python Scripts/train_main.py                                                       # baseline model
    python Scripts/train_main.py --dataset Dataset/4_per_bin/Reduced_103_rot10          # disk rotation
    python Scripts/train_main.py --dataset Dataset/4_per_bin/Reduced_103_dup36          # no rotation
    python Scripts/train_main.py --dataset Dataset/4_per_bin/Reduced_103_dup36 --degrees 180   # Ultralytics rotation
Runs are written to Runs/<dataset folder>/<run name>/: weights/best.pt, and results.csv with the validation Angle MAE
per epoch (metrics/angle_MAE).
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import ultralytics
from ultralytics import YOLO

from angle_fitness import use_val_angle_mae_fitness

REPO_DIR = Path(__file__).resolve().parents[1]
PRETRAINED_MODEL = REPO_DIR / "Weights" / "yolo26n-pose.pt"
PINNED = {"ultralytics": "8.4.157", "torch": "2.14.0"}  # versions the results were produced with


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO26n-pose with the validation Angle MAE as fitness.")
    parser.add_argument("--dataset", default="Dataset/Primary_224px",
                        help="dataset folder holding train/, val/, test/ and one data YAML (relative to the repo root)")
    parser.add_argument("--seed", type=int, default=614, help="training seed (the reported runs use 614-618)")
    parser.add_argument("--degrees", type=float, default=0.0,
                        help="Ultralytics random rotation range in degrees (180 for the Ultralytics rotation model)")
    return parser.parse_args()


def training_device() -> str:
    """Ultralytics device string: GPU 0 if CUDA is available, else CPU (much slower)."""
    if torch.cuda.is_available():
        print(f"Training on GPU 0: {torch.cuda.get_device_name(0)}")
        return "0"
    print("CUDA is not available: training on the CPU")
    return "cpu"


def main() -> None:
    args = parse_args()
    dataset_dir = (REPO_DIR / args.dataset).resolve()
    yamls = sorted(dataset_dir.glob("*.yaml"))
    if len(yamls) != 1:
        sys.exit(f"Expected exactly one data YAML in {dataset_dir}, found {len(yamls)}")

    # Seeds and deterministic kernels, so that a rerun reproduces the same training
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    for name, version in (("ultralytics", ultralytics.__version__), ("torch", torch.__version__)):
        if not version.startswith(PINNED[name]):
            print(f"Warning: {name} {version} differs from {PINNED[name]}, used for the reported results")

    run_name = f"yolo26n seed{args.seed}" + (f" degrees{args.degrees:g}" if args.degrees else "")
    train_args = {
        "data": str(yamls[0]),
        "device": training_device(),
        "epochs": 500,  # upper limit; early stopping ends the runs much earlier
        "patience": 50,  # epochs without a better validation Angle MAE before stopping
        "imgsz": 224,
        "batch": 24,
        "workers": 0,
        "seed": args.seed,
        "deterministic": True,
        "project": str(REPO_DIR / "Runs" / dataset_dir.name),
        "name": run_name,
        "exist_ok": False,
        "save_period": 50,  # keep a checkpoint every 50 epochs
        "optimizer": "AdamW",
        "lr0": 0.001,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "warmup_bias_lr": 0.0,
        "cos_lr": False,
        "val": True,
        "split": "val",
        "amp": True,
        "plots": True,
        "max_det": 1,  # one particle per image
        "iou": 0.7,
        "conf": 0.01,
        # Augmentation stated explicitly instead of inheriting Ultralytics' defaults; only --degrees can turn one on.
        "degrees": args.degrees,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.0,
        "mosaic": 0.0,
        "close_mosaic": 0,
        "mixup": 0.0,
        "cutmix": 0.0,
        "copy_paste": 0.0,
        "erasing": 0.0,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "bgr": 0.0,
        "fraction": 1.0,
        # Loss gains: Ultralytics' defaults, stated for reproducibility
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,
        "pose": 12.0,
        "kobj": 1.0,
        "rle": 1.0,
    }
    for key, value in train_args.items():
        print(f"{key}: {value}")

    use_val_angle_mae_fitness()  # fitness = -validation Angle MAE; square validation canvas
    model = YOLO(str(PRETRAINED_MODEL))
    model.train(**train_args)


if __name__ == "__main__":
    main()
