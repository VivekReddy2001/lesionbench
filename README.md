# LesionBench

[![CI](https://github.com/VivekReddy2001/lesionbench/actions/workflows/ci.yml/badge.svg)](https://github.com/VivekReddy2001/lesionbench/actions/workflows/ci.yml)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

**Does synthetic data actually help an imbalanced skin-lesion classifier, or
does it only look that way?**

HAM10000 is the standard benchmark for dermoscopic image classification:
10,015 images, 7 diagnoses, and a 58:1 imbalance between the largest class
(melanocytic nevi, 67% of images) and the smallest (dermatofibroma). GAN-based
augmentation is a popular fix, including in our own
[SkinAid paper (IEEE OCIT 2021)](https://github.com/VivekReddy2001/SkinLesions_GenerativeAI).
This repository re-asks that question with the controls a skeptical reviewer
would want:

- **Lesion-level splits.** HAM10000 has 10,015 images of only 7,470
  lesions, because many lesions were photographed more than once. With the
  usual image-level random split, **34% of test images have a sibling
  photograph of the same lesion in the training set**. Every headline number
  here uses splits where no lesion appears on both sides. The leaky protocol
  is also run, to measure how much it inflates results.
- **Same compute for every method.** Every method gets the same network,
  optimiser, schedule and number of gradient steps. Methods that enlarge the
  training set (oversampling, GAN top-up) do not get extra epochs.
- **A generator that never sees the test set.** The GAN is retrained for
  every seed on that seed's training split only.
- **Metrics that resist imbalance.** Balanced accuracy and macro-F1 lead,
  plus per-class recall, macro AUROC and expected calibration error.
  Plain accuracy is reported but not used to rank methods: predicting "nevus"
  for everything already scores 67%.
- **A memorisation check.** Are synthetic images new, or near-copies of
  training images?

<!-- RESULTS:START -->
> **Status:** the full benchmark (3 seeds × 8 methods + split comparison) is
> being run with `scripts/run_all.sh`; this section is filled in from
> `results/RESULTS.md` when it finishes.
<!-- RESULTS:END -->

---

## Methods compared

All methods train the same ~1.2M-parameter ResNet-style CNN at 64×64 for
2,000 steps of batch 128 (AdamW, one-cycle LR). The checkpoint with the best
**validation** balanced accuracy is evaluated once on the test split.

| Method | What changes |
|---|---|
| `erm` | Nothing: plain cross-entropy, no augmentation |
| `aug` | Flips, 90° rotations, ±4 px shifts, ±15% brightness/contrast |
| `weighted` | `aug` + cross-entropy weighted by inverse class frequency |
| `focal` | `aug` + focal loss (γ = 2) |
| `oversample` | `aug` + class-balanced sampling (minority images repeat) |
| `gan` | `aug` + each minority class topped up with **synthetic** images to the size of the largest class |
| `gan_mix` | `aug` + class-balanced sampling, where each minority class draws half its samples from real and half from synthetic images |
| `synthetic_only` | Trained on synthetic images only, tested on real ones (a direct measure of synthetic-data quality) |

`oversample` and `gan` are the key pair. Both present a balanced training
stream; the only difference is whether minority examples are *repeated real
images* or *new synthetic ones*. If synthesis adds information, `gan` should
win.

### The generator

A class-conditional **WGAN-GP** (Gulrajani et al., 2017) at 64×64:

- DCGAN-style generator with a learned class embedding concatenated to the
  noise;
- projection critic (Miyato & Koyama, 2018) with layer norm instead of
  batch norm, since the gradient penalty is defined per sample;
- 5 critic steps per generator step, λ = 10, Adam (β = 0, 0.9), lr 1e-4,
  3,000 generator iterations;
- square-root class-balanced sampling, so rare classes are seen more often
  than their frequency but not so often that the critic memorises them;
- only the symmetries dermoscopy genuinely has (rotations, flips) are used
  as augmentation, so no artefacts leak into the generated distribution.

The paper version trained one unconditional generator per class. A single
conditional model shares capacity across classes, which matters when the
smallest class has about 80 training images.

---

## Reproducing

```bash
pip install -e ".[dev]"

# 1. download ISIC 2018 Task 3 (= HAM10000, 2.7 GB) and build a 64x64 .npz (~100 MB)
python -m lesionbench prepare

# 2. everything: 3 seeds x (GAN + 8 methods) + image-level split runs (~6 h on a 4-core CPU)
bash scripts/run_all.sh

# or a single configuration
python -m lesionbench gan --seed 0
python -m lesionbench classify --method gan --split lesion --seed 0

# 3. tables and figures -> results/RESULTS.md, results/figures/
python -m lesionbench report
```

No GPU is needed. Every run writes its configuration, validation and test
metrics, training curve and test-set probabilities to `results/runs/`. The
`report` step rebuilds every table from those files.

The data comes from the official ISIC 2018 Challenge release, including the
`LesionGroupings.csv` file that maps each image to its lesion:
`https://isic-challenge-data.s3.amazonaws.com/2018/`.

## Project layout

```text
lesionbench/
  data.py      download layout, 64x64 preprocessing, lesion- and image-level splits
  models.py    SmallResNet classifier; conditional WGAN-GP generator + projection critic
  gan.py       GAN training on the training split, sampling, memorisation check
  train.py     the eight methods, fixed-step training, checkpoint selection
  metrics.py   balanced accuracy, macro-F1, per-class recall, AUROC, ECE
  report.py    tables and figures from results/runs/*.json
  cli.py       prepare / gan / classify / report
scripts/run_all.sh   the full, resumable benchmark
tests/               fast tests on synthetic data (no download needed)
```

## Limitations

- **64×64 resolution and a small CNN**, so that the whole benchmark runs on a
  CPU. Absolute numbers are well below ImageNet-pretrained models at full
  resolution. The comparison between methods under identical conditions is
  the point, not the absolute ceiling.
- **One generator family.** Diffusion models are a natural next comparison.
- **Pixel-space memorisation check.** Feature-space distances (e.g.
  Inception features) would be more sensitive. Pretrained weights were
  deliberately avoided to keep the pipeline self-contained.

## Related

- Medi, Nemani, **Pitta**, Udutalapally, Das, Mohanty. *SkinAid: A GAN-based
  Automatic Skin Lesion Monitoring Method for IoMT Frameworks.* IEEE OCIT 2021.
  [DOI 10.1109/OCIT53463.2021.00048](https://doi.org/10.1109/OCIT53463.2021.00048) ·
  [code](https://github.com/VivekReddy2001/SkinLesions_GenerativeAI)
- Tschandl, Rosendahl, Kittler. *The HAM10000 dataset.* Scientific Data 5, 2018.

## License

MIT, see [LICENSE](LICENSE). The HAM10000 images are distributed by ISIC under
CC BY-NC 4.0 and are **not** included in this repository.
