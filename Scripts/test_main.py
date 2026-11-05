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

"""Evaluate a trained YOLO26n-pose hinge model on one split: Angle MAE, Angle Bias, error plots, per-particle predictions.

For every particle the Annotated Angle (from the label's keypoints) is compared with the Predicted Angle (from the
model's most confident detection); a missed particle counts as 90° in the Angle MAE, as in training. Outputs, in the
results folder:
    predictions.csv                 one row per particle: angles, Angle Error, keypoints (px), confidences
    comparison_summary.csv          detection rate, Angle MAE, Angle MAE (detected particles), Angle Bias
    angle_error_statistics.png      CDF and histogram of the Angle Error
    error_by_angle_range.png        Angle Error per 10° range of the Annotated Angle
    pred_vs_gt_scatter_box_conf.png Predicted vs Annotated Angle, coloured by Detection confidence
    angle_samples_<range>.png       8 particles per Angle Error range, annotated (green) and predicted (red) keypoints
    run_details.txt                 model, dataset, versions and settings behind the numbers

Usage, from the repo root:
    python Scripts/test_main.py --weights Weights/4_per_bin_disk_rotation/best.pt            # test split
    python Scripts/test_main.py --weights Runs/Primary_224px/<run>/weights/best.pt --split val
The results folder defaults to Results/<model folder>/<split>_<checkpoint>, e.g. Results/4_per_bin_disk_rotation/test_best.
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

from angle_fitness import MISS_PENALTY_DEG

REPO_DIR = Path(__file__).resolve().parents[1]
SEED = 614  # picks the sample particles shown per error range
CONF_THRESHOLD = 0.01  # lowest Detection confidence that counts as a detection, as in training-time validation
SPLIT_NAMES = {"train": "Train", "val": "Validation", "test": "Test"}

PLOT_FONTSIZE = 15
plt.rcParams.update({key: PLOT_FONTSIZE for key in (
    "font.size", "axes.titlesize", "axes.labelsize", "xtick.labelsize", "ytick.labelsize", "legend.fontsize",
    "figure.titlesize")})

# Set in main() from the command line
MODEL_PATH = DATASET_DIR = RESULTS_DIR = None
EVAL_SPLIT = "test"


# ============================================================
# Geometry and labels
# ============================================================
def calculate_angle(kp1: Tuple[float, float], kp2: Tuple[float, float], kp3: Tuple[float, float]) -> float:
    """Angle (deg) at kp2 between the arms to kp1 and kp3."""
    v1 = np.array([kp1[0] - kp2[0], kp1[1] - kp2[1]], dtype=float)
    v2 = np.array([kp3[0] - kp2[0], kp3[1] - kp2[1]], dtype=float)
    denom = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6
    cos_angle = np.clip(np.dot(v1, v2) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos_angle)))


def draw_keypoints_on_image(img: np.ndarray, kp1: Tuple[int, int], kp2: Tuple[int, int], kp3: Tuple[int, int],
                            color: Tuple[int, int, int]) -> None:
    for kp in (kp1, kp2, kp3):
        cv2.circle(img, kp, 2, color, -1)
    cv2.line(img, kp1, kp2, color, 1)
    cv2.line(img, kp2, kp3, color, 1)


def read_annotated_keypoints(label_file: Path, img_w: float, img_h: float):
    """Tip A, vertex, tip B (px) from a one-line YOLO pose label: class, box (4), then x, y per keypoint, each followed by
    its visibility when the dataset's kpt_shape is [3, 3] (14 fields) and not when it is [3, 2] (11 fields)."""
    tokens = label_file.read_text().split()
    if len(tokens) not in (11, 14):
        raise ValueError(f"Unsupported label format with {len(tokens)} fields")
    step = (len(tokens) - 5) // 3
    return tuple((float(tokens[i]) * img_w, float(tokens[i + 1]) * img_h) for i in range(5, len(tokens), step))


def extract_prediction(result) -> Optional[Dict[str, float]]:
    """Keypoints, Predicted Angle and confidences of the most confident detection, or None if nothing was detected."""
    if result.keypoints is None or len(result.keypoints) == 0:
        return None
    kps = result.keypoints.xy[0].cpu().numpy()
    if kps.ndim != 2 or kps.shape[0] < 3:
        return None
    kps_conf = (result.keypoints.conf[0].cpu().numpy().astype(float) if result.keypoints.conf is not None
                else np.full(3, np.nan))  # none from a model trained without visibility (kpt_shape [3, 2])
    box_conf = float(result.boxes.conf[0].cpu()) if result.boxes is not None and len(result.boxes) else np.nan
    return {
        "pred_kp1_x": float(kps[0][0]), "pred_kp1_y": float(kps[0][1]),
        "pred_kp2_x": float(kps[1][0]), "pred_kp2_y": float(kps[1][1]),
        "pred_kp3_x": float(kps[2][0]), "pred_kp3_y": float(kps[2][1]),
        "pred_angle": calculate_angle(tuple(kps[0]), tuple(kps[1]), tuple(kps[2])),
        "box_confidence": box_conf,
        "kp1_confidence": float(kps_conf[0]), "kp2_confidence": float(kps_conf[1]), "kp3_confidence": float(kps_conf[2]),
    }


# ============================================================
# Inference
# ============================================================
def run_inference(model: YOLO, img_dir: Path, label_dir: Path) -> pd.DataFrame:
    """One row per image: annotated and predicted keypoints (px of the source image), angles, Angle Error."""
    rows = []
    stream = model(str(img_dir), stream=True, imgsz=model.overrides["imgsz"], conf=CONF_THRESHOLD)  # training imgsz
    for result in tqdm(stream, desc=f"Inference ({img_dir.name})"):
        img_name = Path(result.path).name
        img_h, img_w = result.orig_shape  # predictions come in source-image pixels; labels are denormalized to match
        try:
            gt_kp1, gt_kp2, gt_kp3 = read_annotated_keypoints(label_dir / f"{Path(result.path).stem}.txt", img_w, img_h)
        except Exception as exc:  # noqa: BLE001
            print(f"Skipping {img_name}: could not parse label ({exc})")
            continue
        row = {
            "image": img_name,
            "gt_kp1_x": gt_kp1[0], "gt_kp1_y": gt_kp1[1], "gt_kp2_x": gt_kp2[0], "gt_kp2_y": gt_kp2[1],
            "gt_kp3_x": gt_kp3[0], "gt_kp3_y": gt_kp3[1],
            "gt_angle": calculate_angle(gt_kp1, gt_kp2, gt_kp3),
            **{key: np.nan for key in ("pred_kp1_x", "pred_kp1_y", "pred_kp2_x", "pred_kp2_y", "pred_kp3_x", "pred_kp3_y",
                                       "pred_angle", "box_confidence", "kp1_confidence", "kp2_confidence",
                                       "kp3_confidence")},
        }
        prediction = extract_prediction(result)
        if prediction is not None:
            row.update(prediction)
        row["detected"] = pd.notna(row["pred_angle"])
        row["angle_error"] = row["pred_angle"] - row["gt_angle"] if row["detected"] else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


PREDICTION_COLUMNS = {  # internal column -> predictions.csv column (kp1 / kp2 / kp3 = tip A / vertex / tip B)
    "image": "image",
    "detected": "detected",
    "gt_angle": "annotated_angle",
    "pred_angle": "predicted_angle",
    "angle_error": "angle_error",
    "box_confidence": "detection_confidence",
    "gt_kp1_x": "annotated_tip_a_x_px",
    "gt_kp1_y": "annotated_tip_a_y_px",
    "gt_kp2_x": "annotated_vertex_x_px",
    "gt_kp2_y": "annotated_vertex_y_px",
    "gt_kp3_x": "annotated_tip_b_x_px",
    "gt_kp3_y": "annotated_tip_b_y_px",
    "pred_kp1_x": "predicted_tip_a_x_px",
    "pred_kp1_y": "predicted_tip_a_y_px",
    "pred_kp2_x": "predicted_vertex_x_px",
    "pred_kp2_y": "predicted_vertex_y_px",
    "pred_kp3_x": "predicted_tip_b_x_px",
    "pred_kp3_y": "predicted_tip_b_y_px",
    "kp1_confidence": "tip_a_keypoint_confidence",
    "kp2_confidence": "vertex_keypoint_confidence",
    "kp3_confidence": "tip_b_keypoint_confidence",
}


def write_predictions(df: pd.DataFrame) -> None:
    """One row per particle (a missed one keeps its Predicted fields empty), 7 significant digits."""
    df[list(PREDICTION_COLUMNS)].rename(columns=PREDICTION_COLUMNS).to_csv(
        RESULTS_DIR / "predictions.csv", index=False, float_format="%.7g")
    print(f"  Saved predictions → {RESULTS_DIR / 'predictions.csv'}")


def write_summary(df: pd.DataFrame) -> None:
    """comparison_summary.csv, one row: detection rate, Angle MAE (a miss counts 90°), the same over the detected
    particles only, and Angle Bias (detected particles)."""
    errors = df.loc[df["detected"], "angle_error"]
    summary = {
        "n_total": len(df),
        "n_detected": len(errors),
        "detection_rate_pct": round(100.0 * len(errors) / len(df), 1),
        "angle_mae_deg": round(df["angle_error"].abs().where(df["detected"], MISS_PENALTY_DEG).mean(), 3),
        "angle_mae_detected_deg": round(errors.abs().mean(), 3),
        "angle_bias_deg": round(errors.mean(), 3),
    }
    pd.DataFrame([summary]).to_csv(RESULTS_DIR / "comparison_summary.csv", index=False)
    print("\n" + "\n".join(f"  {key}: {value}" for key, value in summary.items()))


# ============================================================
# Error plots
# ============================================================
def plot_error_distribution(values: pd.Series) -> None:
    """CDF of the Angle Error with ±5° and ±10° lines (left); share of particles per error range (right):
    < -15, [-15, -5), [-5, 5], (5, 15], > 15."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sorted_errors = np.sort(values.to_numpy())
    axes[0].plot(sorted_errors, 100 * np.arange(1, len(sorted_errors) + 1) / len(sorted_errors), linewidth=2)
    axes[0].axhline(50, color="k", linestyle="--", alpha=0.5, label="Median")
    axes[0].axhline(95, color="g", linestyle="--", alpha=0.5, label="95th percentile")
    axes[0].axhline(5, color="r", linestyle="--", alpha=0.5, label="5th percentile")
    for thresh, color in ((5, "orange"), (10, "purple")):
        axes[0].axvline(-thresh, color=color, linestyle="--", alpha=0.5, label=f"±{thresh}°")
        axes[0].axvline(thresh, color=color, linestyle="--", alpha=0.5)
    axes[0].set(xlabel="Angle Error (°)", ylabel="Cumulative Percentage (%)", title="CDF of Angle Errors")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    lo, hi = 5.0, 15.0
    v = values.to_numpy()
    counts = [int(np.sum(v < -hi)), int(np.sum((v >= -hi) & (v < -lo))), int(np.sum((v >= -lo) & (v <= lo))),
              int(np.sum((v > lo) & (v <= hi))), int(np.sum(v > hi))]
    percentages = [100.0 * c / len(v) for c in counts]
    bars = axes[1].bar([f"< −{hi:g}", f"−{hi:g} to −{lo:g}", f"±{lo:g}", f"{lo:g} to {hi:g}", f"> {hi:g}"], percentages,
                       edgecolor="black", alpha=0.7)
    axes[1].set_ylim(0, 1.25 * max(percentages))  # headroom for the value labels
    axes[1].set(ylabel="Percentage (%)", xlabel="Angle Error (°)", title="Angle Errors Distribution")
    axes[1].grid(True, alpha=0.3, axis="y")
    for bar, pct, count in zip(bars, percentages, counts):
        axes[1].text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{pct:.1f}%\n({count})", ha="center",
                     va="bottom")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "angle_error_statistics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_error_by_range(df: pd.DataFrame) -> None:
    """Angle Error per 10° range of the Annotated Angle [10k, 10k + 10): printed and as a box plot."""
    labels = [f"{i}–{i + 10}" for i in range(0, 180, 10)]
    ranges = pd.cut(df["gt_angle"], bins=list(range(0, 181, 10)), labels=labels, right=False)
    print("\n=== Angle Error per 10° range of the Annotated Angle (detected particles) ===")
    groups = []
    for label in labels:
        subset = df.loc[ranges == label, "angle_error"].dropna()
        if subset.empty:
            continue
        groups.append((label, subset))
        print(f"  {label}: Angle Bias {subset.mean():+.4g} ± {subset.std():.4g}°, Angle MAE {subset.abs().mean():.4g}° "
              f"(n={len(subset)})")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.boxplot([s for _, s in groups], tick_labels=[f"{label}\n({len(s)})" for label, s in groups])
    ax.tick_params(axis="x", labelsize=PLOT_FONTSIZE - 3)
    ax.set(xlabel="Annotated Angle (°) (particles per range)", ylabel="Angle Error (°)", title="Angle Error per 10° range")
    ax.axhline(0, color="red", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "error_by_angle_range.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def error_range_label(min_error: float, max_error: float) -> str:
    if np.isneginf(min_error):
        return f"lt_{max_error:g}"
    if np.isposinf(max_error):
        return f"ge_{min_error:g}"
    return f"{min_error:g}_to_{max_error:g}"


def plot_samples_by_error(df: pd.DataFrame, min_error: float, max_error: float) -> None:
    """Up to 8 detected particles with min_error <= Angle Error < max_error: annotated (green), predicted (red)."""
    selected = df[df["detected"] & (df["angle_error"] >= min_error) & (df["angle_error"] < max_error)]
    if selected.empty:
        print(f"No particles with Angle Error in [{min_error}, {max_error})°")
        return
    sample_df = selected.sample(n=min(8, len(selected)), random_state=SEED)
    print(f"Found {len(selected)} particles with Angle Error in [{min_error}, {max_error})°")
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    for ax in axes:
        ax.axis("off")
    for ax, (_, row) in zip(axes, sample_df.iterrows()):
        img = cv2.imread(str(DATASET_DIR / EVAL_SPLIT / "images" / row["image"]))
        if img is None:
            ax.set_title(f"Unreadable image:\n{row['image']}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for prefix, color in (("gt", (0, 255, 0)), ("pred", (255, 0, 0))):
            draw_keypoints_on_image(img, *[(int(row[f"{prefix}_kp{i}_x"]), int(row[f"{prefix}_kp{i}_y"])) for i in (1, 2, 3)],
                                    color)
        ax.imshow(img)
        ax.set_title(f"Annotated: {row['gt_angle']:.1f}° | Predicted: {row['pred_angle']:.1f}°\n"
                     f"Angle Error: {row['angle_error']:+.2f}°")
    label = error_range_label(min_error, max_error)
    plt.suptitle(f"Angle Error range: {label}°", y=0.98)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"angle_samples_{label}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_predicted_vs_annotated(valid_df: pd.DataFrame, n_particles: int) -> None:
    """Predicted vs Annotated Angle, one dot per detected particle, coloured by Detection confidence (fixed 0-1 scale)."""
    fig, ax = plt.subplots(figsize=(7.5, 6))
    lim = (0, 10 * np.ceil(max(valid_df["gt_angle"].max(), valid_df["pred_angle"].max()) / 10))
    ax.plot(lim, lim, color="0.5", linestyle="--", linewidth=1, label="Predicted = Annotated")
    sc = ax.scatter(valid_df["gt_angle"], valid_df["pred_angle"], c=valid_df["box_confidence"], cmap="Blues", vmin=0,
                    vmax=1, s=20, alpha=0.6, edgecolors="0.2", linewidths=0.3)
    count = (f"{n_particles} particles" if len(valid_df) == n_particles
             else f"{len(valid_df)} of {n_particles} particles detected")
    ax.set(xlim=lim, ylim=lim, aspect="equal", xlabel="Annotated Angle (°)", ylabel="Predicted Angle (°)",
           title=f"{SPLIT_NAMES.get(EVAL_SPLIT, EVAL_SPLIT)} set ({count})")
    ax.grid(True, alpha=0.5)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left")
    plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04).set_label("Detection confidence")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "pred_vs_gt_scatter_box_conf.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_run_details(model: YOLO, df: pd.DataFrame, valid_df: pd.DataFrame) -> None:
    """The model, dataset, package versions and settings behind the numbers in this folder."""
    import torch
    import ultralytics

    def rel(path: Path) -> str:
        path = path.resolve()
        return str(path.relative_to(REPO_DIR)) if path.is_relative_to(REPO_DIR) else str(path)

    train_args = (getattr(model, "ckpt", None) or {}).get("train_args", {}) or {}
    lines = [
        f"Written: {datetime.now().isoformat(timespec='seconds')}",
        f"Model: {rel(MODEL_PATH)}",
        "Checkpoint training args: " + ", ".join(f"{k}={train_args[k]}" for k in ("imgsz", "epochs", "degrees", "seed")
                                                 if k in train_args),
        f"Dataset: {rel(DATASET_DIR)}, split {EVAL_SPLIT}",
        f"Inference: imgsz {model.overrides['imgsz']} (the training imgsz), conf {CONF_THRESHOLD}, fp32",
        f"Python {sys.version.split()[0]}, ultralytics {ultralytics.__version__}, torch {torch.__version__}, "
        f"device {'cuda' if torch.cuda.is_available() else 'cpu'}",
        f"Particles: {len(df)}, detected: {len(valid_df)} ({100.0 * len(valid_df) / max(len(df), 1):.1f}%)",
    ]
    (RESULTS_DIR / "run_details.txt").write_text("\n".join(lines) + "\n")


# ============================================================
# Main
# ============================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a YOLO26n-pose hinge model on one split.")
    parser.add_argument("--weights", default="Weights/4_per_bin_disk_rotation/best.pt",
                        help="checkpoint to evaluate (relative to the repo root)")
    parser.add_argument("--dataset", default="Dataset/Primary_224px",
                        help="dataset folder holding <split>/images and <split>/labels (relative to the repo root)")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--out", default=None, help="results folder (default: Results/<model folder>/<split>_<checkpoint>)")
    return parser.parse_args()


def main() -> None:
    global MODEL_PATH, DATASET_DIR, RESULTS_DIR, EVAL_SPLIT
    args = parse_args()
    MODEL_PATH, DATASET_DIR, EVAL_SPLIT = (REPO_DIR / args.weights).resolve(), (REPO_DIR / args.dataset).resolve(), args.split
    model_dir = MODEL_PATH.parent.parent if MODEL_PATH.parent.name == "weights" else MODEL_PATH.parent  # Runs/.../<run>/weights/
    RESULTS_DIR = (REPO_DIR / args.out).resolve() if args.out else \
        REPO_DIR / "Results" / model_dir.name / f"{EVAL_SPLIT}_{MODEL_PATH.stem}"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Model: {MODEL_PATH}\nDataset: {DATASET_DIR} ({EVAL_SPLIT})\nResults: {RESULTS_DIR}")

    model = YOLO(MODEL_PATH)
    df = run_inference(model, DATASET_DIR / EVAL_SPLIT / "images", DATASET_DIR / EVAL_SPLIT / "labels")
    if df.empty:
        sys.exit("No particles processed.")
    valid_df = df[df["detected"]].copy()
    print(f"\nParticles: {len(df)}, detected: {len(valid_df)} ({100.0 * len(valid_df) / len(df):.1f}%)")

    write_run_details(model, df, valid_df)
    write_predictions(df)
    write_summary(df)
    if valid_df.empty:
        sys.exit("No detections: skipping the error plots.")
    plot_error_distribution(valid_df["angle_error"])
    plot_error_by_range(valid_df)
    for min_e, max_e in ((-5, 5), (-10, -5), (5, 10), (-np.inf, -10), (10, np.inf)):
        plot_samples_by_error(valid_df, min_e, max_e)
    plot_predicted_vs_annotated(valid_df, n_particles=len(df))
    print(f"\nAll outputs saved in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
