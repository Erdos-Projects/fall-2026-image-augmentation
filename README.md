# Rotation Is All You Need: hinge angles from few annotated particles

Deep-learning models that measure DNA-origami pose-estimation in TEM images need many annotated particles (>1000 images). This project
measures the vertex angle of a DNA-origami hinge (a nano-caliper: two arms joined at a vertex) with a YOLO26n-pose model
that predicts three keypoints per particle (tip A, vertex, tip B), and asks how few annotated particles it needs when
rotation augmentation turns each of them into many training images.

The answer on the hinge data of Wang, Jin & Castro (2023): **103 annotated particles (37.9× fewer than the full 3907)
give a test Angle MAE of 3.04°, close to the 2–3° human floor**; the model trained on all 3907 particles gives 2.65°.

## Results

Test set: 489 particles, never used for training or model selection.

| model | training particles | Angle MAE, seed 614 (shipped) | Angle MAE, mean ± SD over 5 training seeds |
|---|---|---|---|
| baseline model | 3907 | 2.69° | 2.65 ± 0.09° |
| 4 per bin, disk rotation | 103 (37.9× fewer) | 3.08° | 3.04 ± 0.11° |
| 4 per bin, Ultralytics rotation | 103 | 3.45° | 3.34 ± 0.10° |
| 4 per bin, no rotation | 103 | 11.72° | (seed 614 only) |

- Without rotation, 103 particles are far from enough; rotation makes up the difference.
- Disk rotation beats Ultralytics' built-in rotation, which turns the whole image, cuts off-center hinges and fills the
  corners with flat grey.
- The human floor is the error that annotation alone carries: 2–3° Angle MAE (two annotators differ by 3.05° in the
  source publication).
- Detection rate 100% for every model with rotation (99.8% without).
- Cost on one 8 GB GPU: 1.0–2.1 h per 4-per-bin training run, 1.5–3.0 h for the baseline model; inference 11.5 ms per
  particle. Each model trained for 500 epochs with 50 epochs early stopping and batch size of 24, these can be adjusted if faster training is needed.

`results.ipynb` rebuilds the table and the figures from the shipped predictions, without a GPU or the dataset.

## Approach

1. **Data** (`guidance.md`): the 4884 hinge particles of the source publication, 200 × 200 px TEM crops with three
   annotated keypoints each, converted to 8-bit, 224 px YOLO pose data and split 3907 / 488 / 489 (train / validation /
   test) with a fixed seed.
2. **Evenly spaced picks**: per 5° bin of the vertex angle, the training particles nearest to 4 evenly spaced target
   angles: 103 particles covering 10–141° (`samples_4_per_bin.csv`, `data_processing/evenly_spaced_picks.png`).
3. **Disk rotation**: each picked particle is turned in 10° steps (36 copies). Only a disk around the hinge turns
   (cubic interpolation, Poisson blending at its edge), so arms are never cut and no padding appears
   (`data_processing/disk_rotation_and_seams.png`).
4. **Training** (`Scripts/train_main.py`): YOLO26n-pose from COCO-pose weights, single-channel 224 px input, AdamW. The
   validation Angle MAE is the fitness: it picks the checkpoint and stops training 50 epochs after the best one (at
   most 500 epochs). Built-in augmentation is off, except Ultralytics' rotation for that condition.
5. **Evaluation** (`Scripts/test_main.py`): Annotated vs Predicted Angle for every test particle.

## Repository layout

```
README.md                   this file
guidance.md                 data download and preparation (YOLO pose format), and how the 4-per-bin sets are built
samples_4_per_bin.csv       the 103 training particles of the 4-per-bin models
requirements.txt            pinned package versions
results.ipynb               results table and figures from the shipped predictions
Scripts/
    train_main.py           training (--dataset, --seed, --degrees)
    test_main.py            evaluation of one checkpoint on one split
    angle_fitness.py        validation Angle MAE as the Ultralytics fitness
Weights/<model>/            best.pt (seed 614), args.yaml (training settings), results.csv (validation per epoch)
Results/
    <model>/test_best/      predictions.csv (one row per test particle), comparison_summary.csv
    per_seed_test_angle_mae.csv   test Angle MAE of every training seed
data_processing/            the data and augmentation steps as figures
LICENSE                     GNU GPL v3
```

`<model>` is `baseline`, `4_per_bin_disk_rotation`, `4_per_bin_ultralytics_rotation` or `4_per_bin_no_rotation`.

## Setup

Python 3.10 and the versions in `requirements.txt` (torch first, built for your CUDA version or the CPU):

```bash
python3.10 -m venv venv && source venv/bin/activate
pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

All commands run from the repo root. Training uses GPU 0 when CUDA is available and the CPU otherwise (much slower).

## Usage

Prepare `Dataset/` as described in `guidance.md`, then:

```bash
# Train (runs go to Runs/<dataset folder>/<run name>/; the COCO-pose weights download on the first run)
python Scripts/train_main.py                                                        # baseline model
python Scripts/train_main.py --dataset Dataset/4_per_bin/Reduced_103_rot10           # 4 per bin, disk rotation
python Scripts/train_main.py --dataset Dataset/4_per_bin/Reduced_103_dup36           # 4 per bin, no rotation
python Scripts/train_main.py --dataset Dataset/4_per_bin/Reduced_103_dup36 --degrees 180   # Ultralytics rotation
# add --seed 615 (616, 617, 618) for the other training seeds

# Evaluate a checkpoint on the test split (outputs in Results/<model>/test_best/)
python Scripts/test_main.py --weights Weights/4_per_bin_disk_rotation/best.pt
python Scripts/test_main.py --weights "Runs/Primary_224px/yolo26n seed614/weights/best.pt"

# Results table and figures
jupyter notebook results.ipynb
```

Rerunning `test_main.py` on a shipped checkpoint reproduces its shipped `predictions.csv` exactly.

## Terms

- **Vertex angle**: the angle at the vertex between the hinge's two arms (tip A – vertex – tip B).
- **Annotated Angle / Predicted Angle**: the vertex angle from a particle's annotated keypoints / from the model's
  keypoints. **Angle Error** = Predicted − Annotated Angle.
- **Angle MAE**: mean |Angle Error| over a split; a missed particle counts as 90°. **Angle Bias**: mean Angle Error.
- **Detection rate**: share of particles in which the model finds the particle with its three keypoints.
- **Human floor**: the Angle MAE that annotation alone carries, 2–3°.
- **Angle bin**: a 5° interval of the vertex angle. **4 per bin**: 4 training particles per angle bin.
- **Rotation disk**: a circle around a hinge's outline plus a 10 px margin; rotating only its pixels re-orients the
  particle without cutting the arms or padding the image. **Rotated copy**: one turn of a particle's disk.
- **Disk rotation / no rotation / Ultralytics rotation**: the three ways the 4-per-bin training images are made:
  rotated copies, 36 unrotated repeats, or the repeats turned by Ultralytics as a whole image (random angle).

## Data source

Wang Y, Jin X, Castro C. *Accelerating the characterization of dynamic DNA origami devices with deep neural networks.*
Sci. Rep. 13, 15196 (2023). [doi:10.1038/s41598-023-41459-w](https://doi.org/10.1038/s41598-023-41459-w).
Data: [OSF, doi:10.17605/OSF.IO/BMXHF](https://doi.org/10.17605/OSF.IO/BMXHF).

## License

GNU General Public License v3 (`LICENSE`). The models are trained with Ultralytics YOLO26, which is licensed under
AGPL-3.0.
