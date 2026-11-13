# Task 4 — A better leaf classifier

*Draft written by the coding agent from the measured runs, to be rewritten in my own
words before hand-in. Every number in here is a measured run from
[`task4/results.csv`](task4/results.csv); nothing is estimated. Rows marked
`… running` had not finished while this draft was written and will be filled in,
not guessed.*

## 1. Introduction

The model we exported in Task 2 runs on the phone, but it was trained on
96 × 96 indoor close-ups that the teacher had already cropped, while the phone
camera shows a leaf somewhere in a 4:3 frame, tilted, in bad light. The question
of this task is which changes actually close that gap, and which only make the
validation number prettier.

I kept the Task 2 pipeline as it is and changed one thing per experiment:
colour and geometric augmentation, a frozen backbone, a learning-rate schedule
and more epochs, five small training knobs (dropout, global average pooling,
AdamW, label smoothing, shipping the best epoch instead of the last one), and
last year's photographs. Each experiment runs with three seeds, because one
validation picture is worth 1.6 % of accuracy and one run says almost nothing.

## 2. Experimental methodology

**Baseline.** The Task 2 network exactly as demonstrated: `mobilenet_v3_small`
pretrained on ImageNet, last layer replaced by 4 outputs, every layer trained,
`Adam(lr=1e-3)`, batch 32, `CrossEntropyLoss`, 15 epochs, no augmentation, the
weights of the **last** epoch. That is the recipe of `leaf_model.onnx`, which is
the file on the phone now.

**Data.** `data/leaves/train` 244 photos and `data/leaves/val` 62 photos
(apple 68/17, cherry 75/19, chestnut 41/11, maple 60/15), all already 96 × 96,
split so that one student's photos of one species stay on one side. The
validation set is **never** modified, re-split or augmented. Last year's set
(`data/leaves_historical`, 175 train + 30 val per species, other students, other
phones) is added only in the rows that say so; I checked that not one of its 820
files is byte-identical to any of our 306, so it is a genuine outside test.

**Three measuring sets, identical for every row.**

| set | n | what it is | what it asks |
| --- | --- | --- | --- |
| `val` | 62 | the group validation photos | did it learn the species, or the photograph? |
| `camera` | 62 | the *same* photos, randomly cropped (45–100 %), rotated ±35°, colour-jittered (brightness/contrast/saturation 0.4, hue 0.08) and blurred, saved once with seed 1234 | will it survive a real phone frame? |
| `lastyear` | 120 | last year's validation photos | leaves nobody in our group photographed |

The camera-like set was generated **once**, written to `data/val_camera/`, and
every model after that only reads it — so all rows are scored on pixel-identical
pictures. For the rows trained on last year's photos, the `lastyear` column is
contaminated and is marked as such.

**Fairness.** Every row is produced by the same `run()` function (notebook
cell 66), which contains the 2c/3c training loop; a row differs from the
baseline only in the argument it passes. Each run calls `torch.manual_seed(seed)`
and builds its training loader with `generator=torch.Generator().manual_seed(seed)`,
so two rows with the same seed see the same weights *and* the same batch order.
The seed cannot make the augmentation stream identical between an augmented and
a non-augmented row — an augmented row draws random numbers that the other does
not — which is exactly why the important rows are repeated over three seeds and
reported with their spread. All runs are on this laptop (14 cores, no GPU),
`torch 2.15.0+cpu`, and the version is recorded in the CSV.

**Not comparable to the notebook's printed numbers.** The notebook was run in
Colab on a GPU with `torch 2.14.1`; it printed 0.823 at epoch 1 and 0.661 at
epoch 15 for this same recipe. My baseline measured 0.769 (mean over three
seeds) at the last epoch. Different device, different random streams. Every row
below is therefore compared **only** against the baseline row measured here,
never against the notebook's numbers. I re-ran the notebook's own cells on this
machine first and they reproduce its printed shapes and counts exactly
(244/62, the same `classifier`, 208 292 parameters), so it is the same pipeline.

## 3. Results

Spread in brackets is max − min over the seeds. One validation picture is worth
0.016, so a spread larger than that means a single run proves nothing.

| Experiment | Change vs baseline | Seeds | Val last epoch | Val best epoch | Camera-like | Last year | Train s |
|---|---|---|---|---|---|---|---|
| `baseline` | none (the deployed recipe) | 0,1,2 | 0.769 (0.113) | 0.823 (0.065) | 0.694 (0.032) | 0.517 (0.067) | 47 |
| `A_colorjitter` | + ColorJitter | 0,1,2 | 0.801 (0.081) | 0.849 (0.048) | 0.737 (0.097) | 0.586 (0.050) | 60 |
| `A2_geometric` | + RandomResizedCrop/flips/rotation (3d) | 0,1,2 | 0.817 (0.032) | 0.849 (0.016) | 0.677 (0.113) | 0.583 (0.117) | 51 |
| `A3_geom+color` | + both augmentations | 0,1,2 | **0.839** (0.048) | 0.860 (0.016) | **0.769** (0.129) | **0.628** (0.100) | 64 |
| `B_frozen_backbone` | only the new head trains | 0,1,2 | 0.790 (0.048) | 0.817 (0.016) | 0.629 (0.065) | 0.464 (0.042) | 14 |
| `C_aug+cosine_40ep` | augmentation + cosine LR + 40 epochs | … running | … | … | … | … | … |
| `D1_dropout0.5` | head dropout 0.2 → 0.5 | … running | … | … | … | … | … |
| `D2_leafnet_flatten` | our Task 3 network, flattened | … running | … | … | … | … | … |
| `D2b_leafnet_gap` | the same network, global average pooling | … running | … | … | … | … | … |
| `D3_adamw_wd0.05` | Adam → AdamW, weight decay 0.05 | … running | … | … | … | … | … |
| `D4_label_smooth0.1` | label smoothing 0.1 | … running | … | … | … | … | … |
| `D5_best_epoch` | ship the best epoch instead of the last | … running | … | … | … | … | … |
| `E_lastyear_only` | trained only on last year's 700 | … running | … | … | … | … | … |
| `E_group+lastyear` | our 244 + last year's 700 | … running | … | … | … | … | … |
| `leaf_model.onnx` | the deployed file, scored with onnxruntime | — | … running | — | … running | … running | — |
| `F_best` | the best combination, decided after the rows above | not started | | | | | |

## 4. Analysis

### 4.1 The seeds vary more than most changes

The baseline alone moves over 0.113 of validation accuracy between its three
seeds (0.823, 0.774, 0.710) — larger than the difference between the baseline
and most of the things I tried. Every row where the spread overlaps the
difference from the baseline is therefore reported as "no clear effect", even
when the mean looks better. On three seeds this design can tell an effect of
about 0.05 apart from noise, and nothing smaller; I say so instead of
interpreting a 0.01.

### 4.2 The last epoch is not the one you want

For **all three** baseline seeds the best epoch was better than the last epoch,
by 0.016, 0.032 and 0.113 (mean 0.823 against 0.769). Seed 0 peaked at epoch 14
(0.855) and closed at 0.823; the notebook's Colab run peaked at epoch 1 and
closed at 0.661. The peak is not at a predictable epoch, so there is nothing
else to do than keep the weights of the best epoch — but the `val best epoch`
column is **chosen on the validation set**, so it flatters itself: it is not an
achievable accuracy, only the honest `D5_best_epoch` row tells what shipping it
actually gives.

### 4.3 Augmentation helps, and colour is the part that transfers

Colour jitter moves the camera-like set from 0.694 to 0.737 and last year's
photos from 0.517 to 0.586; geometric augmentation alone does not help the
camera-like set at all (0.677, and with a 0.113 spread) while it is the row with
the steadiest validation (spread 0.032). Both together is the best row so far on
all three sets (0.839 / 0.769 / 0.628). The reading that fits the numbers: our
validation photos differ mainly in *light and colour*, not in framing — so
jittering colours teaches something the measuring sets reward, whereas cropping
and rotating mostly teaches framing that the (already centre-cropped) validation
photos never ask for. The camera-like set does include crops and rotation, and
geometric augmentation still does not win there, which makes me distrust the
camera-like set as a perfect stand-in for the real camera (see §6).

### 4.4 Freezing the backbone is the cheap option, and it is worse

Training only the new head is 3–4× faster (14 s against 47 s) but it is the
weakest row on both outside measures (0.629 camera-like, 0.464 last year, against
0.694 and 0.517 for the baseline). With only 244 photos the head alone cannot
adapt ImageNet's features to leaves; fine-tuning the whole network is what buys
the transfer. This is the one clean negative result so far, and it is the
opposite of what transfer learning usually suggests on small data — worth
pointing out in class.

### 4.5 Nothing generalises yet

The honest summary of the deployed recipe: 0.769 on our own validation photos,
0.694 on the camera-like versions of them, and 0.517 on other students' leaves —
where guessing would give 0.25. The 0.517 is the number closest to what the
phone will show, and even the best row so far only lifts it to 0.628.

*(sections 4.6–4.9 — the training knobs, the learning-rate schedule, last year's
data and the combination — are written after their rows finish)*

## 5. Final recommendation

*To be written once the remaining rows are measured. It will be argued from the
table on four grounds — validation accuracy, robustness (camera-like and last
year), model size and inference speed on the phone — and it will state which
model I would put on the phone and what I would not claim about it.*

## 6. What is measured and what is assumed

**Measured:** every number in the table, on this machine, from
`task4/results.csv` (42 runs: 14 experiments × 3 seeds, one run per row).

**Assumed, and said so:**
- *The camera-like set stands for a phone camera.* It is a fixed, seeded
  transformation of the validation photos — crop, tilt, colour shift, blur,
  JPEG. A real frame also has motion blur, focus failure, a leaf overlapping
  another leaf, and the app's own central-square crop. Treat it as an
  ordering of models, not as an accuracy you would see on the phone.
- *`lastyear` measures generalisation* — true only for rows that never trained
  on it.
- *Training seconds mean anything for the phone.* They do not: they are CPU
  training times on a 14-core laptop with no GPU, and they say nothing about
  inference speed in the browser. Inference speed will be measured separately.

**Not tested:** the model has **not** been loaded into the course camera app or
into our own page. No claim about the phone is made anywhere in this report
until that test has been run and its screenshot is here.

## 7. How to reproduce

```bash
# once: the two datasets (the notebook's own links) into data/, weights stay out of git
python -m pip install torch torchvision onnx onnxscript     # CPU build is enough

# the whole table, resumable: a rerun skips every (experiment, seed) already in results.csv
jupyter notebook Copy_of_ex08_student.ipynb      # run Task 2's cells, then cells 65-67
```

The camera-like set is regenerated with the same seed 1234 if `data/val_camera`
is missing, so the table stays reproducible from a clean checkout. `task4/log.txt`
has every epoch of every run (loss and validation accuracy), which is where the
"the last epoch is not the best epoch" claim in §4.2 comes from.
