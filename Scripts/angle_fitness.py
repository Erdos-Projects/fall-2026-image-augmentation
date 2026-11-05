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

"""Validation Angle MAE as the training fitness for Ultralytics pose models (written against ultralytics 8.4.157).

use_val_angle_mae_fitness() patches the pose validator so that every validation logs metrics/angle_MAE (-> results.csv)
and replaces Ultralytics' fitness (box + pose mAP) by -Angle MAE, so early stopping and best.pt follow the project KPI.
Validation runs on the square imgsz canvas that prediction uses.
"""
import numpy as np
import torch
from ultralytics.data.base import BaseDataset
from ultralytics.models.yolo.pose import PoseValidator
from ultralytics.utils import LOGGER

MISS_PENALTY_DEG = 90.0  # error charged to an image whose particle is not detected


def square_val_letterbox() -> None:
    """Validate on the square imgsz canvas that prediction (and test_main.py) uses.

    By default Ultralytics validates with rect batches padded by half a stride (a 224 px image becomes a 256 px canvas),
    which shifts the validation Angle MAE against prediction on the same checkpoint.
    """
    init = BaseDataset.__init__

    def __init__(self, *args, **kwargs):
        if not kwargs.get("augment", True):  # validation/test datasets only
            kwargs["rect"] = False
        init(self, *args, **kwargs)

    BaseDataset.__init__ = __init__


def vertex_angle(kpts: torch.Tensor) -> float:
    """Angle (deg) at the vertex (row 1) between the two arms to tip A (row 0) and tip B (row 2)."""
    v1 = (kpts[0, :2] - kpts[1, :2]).double()
    v2 = (kpts[2, :2] - kpts[1, :2]).double()
    cos = torch.dot(v1, v2) / (v1.norm() * v2.norm() + 1e-6)
    return float(torch.rad2deg(torch.arccos(cos.clamp(-1.0, 1.0))))


def use_val_angle_mae_fitness() -> None:
    """Patch PoseValidator: fitness = -mean |Angle Error| of the most confident prediction per image, on the square
    validation canvas. A second call does nothing."""
    if getattr(PoseValidator, "_angle_mae_patched", False):
        return
    PoseValidator._angle_mae_patched = True
    square_val_letterbox()
    init_metrics, process_batch, get_stats = (
        PoseValidator.init_metrics, PoseValidator._process_batch, PoseValidator.get_stats
    )

    def _init_metrics(self, model):
        init_metrics(self, model)
        self.angle_errors = []

    def _process_batch(self, preds, batch):
        # predictions and annotated keypoints share the input pixel frame; predictions are sorted by confidence
        if batch["keypoints"].shape[0]:
            annotated = vertex_angle(batch["keypoints"][0])
            detected = preds["cls"].shape[0] > 0
            self.angle_errors.append(abs(vertex_angle(preds["keypoints"][0]) - annotated) if detected else MISS_PENALTY_DEG)
        return process_batch(self, preds, batch)

    def _get_stats(self):
        stats = get_stats(self)
        mae = float(np.mean(self.angle_errors)) if self.angle_errors else float("nan")
        stats["metrics/angle_MAE"] = mae
        stats["fitness"] = -mae  # Ultralytics maximizes fitness
        LOGGER.info(f"val angle MAE: {mae:.3f} deg over {len(self.angle_errors)} images")
        return stats

    PoseValidator.init_metrics = _init_metrics
    PoseValidator._process_batch = _process_batch
    PoseValidator.get_stats = _get_stats
