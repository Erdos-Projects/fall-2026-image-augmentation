# Rotation Is All You Need: executive summary

## Problem
Labs building dynamic DNA-origami devices measure their conformations in TEM images, particle by particle; for a hinge
(nano-caliper: two arms joined at a vertex), that is the vertex angle. Deep-learning pose models automate this but
typically need >1000 hand-annotated particles per device design to approach the human floor (2–3°, the error of
annotation itself; Wang, Jin & Castro 2023). KPIs (set at the start): ≥10× fewer annotated training particles, with an
error competitive with the full-data model and close to the human floor.

## Approach
1. **Evenly spaced picks**: 4 per 5° angle bin, nearest to evenly spaced angles: 103 of 3907, 10–141°.
2. **Disk rotation**: 36 copies per pick, 10° apart, turning only a disk around the particle.
3. **Model**: YOLO26n-pose (2.7 M parameters, pre-trained on human poses) finds tip A, vertex and tip B.

## Results
Test Angle MAE (mean absolute angle error over 489 particles never used in training or model selection), mean ± SD
over 5 training seeds (no rotation: 1 seed). Baseline model (all 3907 particles): 2.65 ± 0.09°.

| annotated training particles | disk rotation | Ultralytics rotation (whole image) | no rotation |
|:---|:---:|:---:|:---:|
| 28 (139.5× fewer) | 3.91 ± 0.28° | 4.47 ± 0.21° | 33.1° |
| 54 (72.4× fewer) | 3.38 ± 0.24° | 3.86 ± 0.18° | 23.0° |
| **103 (37.9× fewer)** | **3.04 ± 0.11°** | 3.34 ± 0.10° | 11.7° |
| 197 (19.8× fewer) | 2.68 ± 0.09° | 3.13 ± 0.08° | 6.1° |

- KPIs met: 103 particles give 3.04°, close to the human floor, 0.39° above the baseline; 197 match it.
- Without rotation, small sets fail; disk rotation beats the built-in rotation at every size (by 0.30–0.55°).

## Impact for stakeholders (DNA-origami groups, other TEM users)
- **Annotation budget** for a new device: 4 particles per 5° bin of its angle range for near-human accuracy (103 here);
  8 per bin to match the model trained on all 3907 (197 here).
- **Augmentation**: rotate the particle (physically guided), not the whole image, which cuts off-center arms and adds
  grey corners.
- **Picking**: random picks gave the same overall error (3.03–3.07° vs 3.08° for evenly spaced picks, one run each);
  the model interpolates to unseen angles whatever the picking approach.
- **Practical**: one 8 GB GPU, 1.0–2.1 h per training run, 11.5 ms per particle; detection rate 100%. Training can be
  sped up as needed (epochs, batch size and early stopping were not tuned).
- **Limits**: one device and one dataset tested, so other designs may need other counts; only the training set was
  reduced (validation 488 and test 489 unchanged).
- **Next**: smaller validation sets; other designs; more seeds and draws; varied picks (scale, brightness/contrast);
  fewer/random rotation angles; interpolation/extrapolation limits; training settings.

Data: Wang, Jin & Castro, Sci. Rep. 13, 15196 (2023); [OSF, doi:10.17605/OSF.IO/BMXHF](https://doi.org/10.17605/OSF.IO/BMXHF).
Code, weights and per-particle results: this repository (`README.md`).
