# Data preparation

The models are trained and tested on the DNA-origami hinge particles of Wang, Jin & Castro, *Accelerating the
characterization of dynamic DNA origami devices with deep neural networks*, Sci. Rep. 13, 15196 (2023),
[doi:10.1038/s41598-023-41459-w](https://doi.org/10.1038/s41598-023-41459-w). Their TEM particle images and keypoint
annotations are public on OSF: [doi:10.17605/OSF.IO/BMXHF](https://doi.org/10.17605/OSF.IO/BMXHF). This guide turns
them into the YOLO pose datasets that `Scripts/train_main.py` and `Scripts/test_main.py` read.

## 1. Download

OSF → Files → `Dataset/Dataset.zip` (1.96 GB). From the zip, use:

- `Dataset/nDFS_Dataset/Image Particle Dataset/`: one 200 × 200 px, 16-bit grayscale PNG per particle (6372 files);
- `Dataset/nDFS_Dataset/Image Particle Dataset_Annotated/<gallery>_Pose.csv`: one row per particle, with the file name
  (`Image_indices`), the two tips (`x1, y1` and `x2, y2`), the vertex (`cross1, cross2`), all in pixels, and the
  vertex angle (`Angle`).

The 20 pose CSVs used, 4884 particles in total:

```
nDFSB_R2_Pose.csv      nDFSC10_g4_Pose.csv    nDFSC20_g1_Pose.csv    nDFSC5_g1_Pose.csv
nDFSB_R2v2_Pose.csv    nDFSC10_g5_Pose.csv    nDFSC20_g2_Pose.csv    nDFSC5_g2_Pose.csv
nDFSC10_g1_Pose.csv    nDFSC15_g1_Pose.csv    nDFSC25_g1_Pose.csv    nDFSC5_g3_Pose.csv
nDFSC10_g2_Pose.csv    nDFSC15_g2_Pose.csv    nDFSC25_g2_Pose.csv    nDFSC5_g4_Pose.csv
nDFSC10_g3_Pose.csv    nDFSC15_g3_Pose.csv    nDFSC30_g1_Pose.csv    nDFSC30_g2_Pose.csv
```

`nDFSB_R3_pred_Pose.csv` (1488 particles) is left out: its images are too blurry.

## 2. Labels (YOLO pose format)

One text file per image (`<image name>.txt`), one line: `0 xc yc w h  xA yA 2  xV yV 2  xB yB 2`.
Keypoints: pixel coordinates ÷ 200, in the order tip A, vertex, tip B (tips counter-clockwise, swapped where needed),
  each followed by visibility 2, which you can leave out (then set `kpt_shape: [3, 2]` in the YAML).
The box is the keypoints' bounds ± 0.1, clipped to [0, 1] (`xc yc` its center, `w h` its size).

Check if needed: the vertex angle from your labels matches the CSV's `Angle` column, and the `annotated_angle` column of the
shipped `Results/*/test_best/predictions.csv` for the test particles.

## 3. Images

- 16 → 8 bit by dropping the low byte (`value >> 8`; `cv2.imread(path, cv2.IMREAD_GRAYSCALE)` gives the same values).
- Resize 200 → 224 px with bilinear interpolation (`cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)`).
  The normalized labels stay valid.
- Save as single-channel PNG with the same file name.

## 4. Split

Fixed random 80 / 10 / 10 split, unstratified:

```python
import numpy as np

stems = sorted(stems)  # the 4884 file names without .png
order = np.random.default_rng(614).permutation(len(stems))
train = [stems[i] for i in order[:3907]]
val = [stems[i] for i in order[3907:3907 + 488]]
test = [stems[i] for i in order[3907 + 488:]]  # 489
```

Check if needed: your test split holds exactly the 489 file names of the shipped `predictions.csv` files.

## 5. Folder layout and data YAML

```
Dataset/Primary_224px/
    primary_224px.yaml
    train/images/  train/labels/
    val/images/    val/labels/
    test/images/   test/labels/
```

`primary_224px.yaml` (no `path:` key, so Ultralytics uses the YAML's folder as the dataset root):

```yaml
train: train/images
val: val/images
test: test/images

names:
  0: hinge

kpt_shape: [3, 3]
channels: 1
skeleton:
  - [0, 1]
  - [1, 2]
```

This is the baseline model's dataset (3907 / 488 / 489 particles). Train with `python Scripts/train_main.py`.

## 6. The 4-per-bin training sets (described here; their code is not part of this repo)

The three 4-per-bin models train on 103 of the 3907 training particles, listed in `samples_4_per_bin.csv`.
Every set keeps the validation and test folders of `Primary_224px` (copy or link them), and its own YAML as above.

- **Evenly spaced picks**: in each 5° angle bin [5k, 5k + 5) of the vertex angle, the particle closest to each of the
  target angles 5k + 1.25°, 5k + 3.75°, 5k + 2.5° and 5k (filled in that order, a particle never picked twice). Bins with fewer
  usable particles give what they have, so the 28 bins (10–141°) hold 103 samples.
- **Rotation disk**: a circle around the hinge's outline plus a 10 px margin. Only particles whose disk fits inside
  the frame can be picked (3759 of the 3907).
- **Disk rotation** (`Dataset/4_per_bin/Reduced_103_rot10/`, 3708 training images): each sample turned clockwise by
  0, 10, …, 350° (36 rotated copies; the 0° copy is the original). Only the pixels inside the disk turn, with cubic
  interpolation, and Poisson blending (OpenCV `seamlessClone`) hides the seam at the disk edge; the pixels outside stay
  in place. The keypoints turn with the same rotation, and the box is rebuilt from them (bounds ± 0.1).
- **No rotation** (`Dataset/4_per_bin/Reduced_103_dup36/`, 3708 training images): each sample repeated 36 times,
  unrotated.
- **Ultralytics rotation**: the no-rotation set trained with `--degrees 180` (Ultralytics turns the whole image by a
  random angle each time it loads it, and fills the corners with grey).

`data_processing/` shows the picks, the disk and its seam, and the rotated samples.
