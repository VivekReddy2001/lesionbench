## Main results: lesion-level split

Test set, mean ± std over seeds. Checkpoint chosen by validation balanced accuracy.

| Method | Bal. acc | Macro-F1 | Accuracy | Macro AUROC | ECE ↓ | Seeds |
|---|---:|---:|---:|---:|---:|---:|
| Baseline (no augmentation) | 0.436 ± 0.022 | 0.436 ± 0.017 | 0.702 ± 0.013 | 0.868 ± 0.016 | 0.208 ± 0.006 | 3 |
| Classic augmentation | 0.554 ± 0.013 | 0.578 ± 0.015 | 0.785 ± 0.013 | 0.939 ± 0.004 | 0.034 ± 0.010 | 3 |
| + class-weighted loss | 0.629 ± 0.017 | 0.510 ± 0.007 | 0.656 ± 0.002 | 0.921 ± 0.006 | 0.064 ± 0.011 | 3 |
| + focal loss | 0.547 ± 0.021 | 0.572 ± 0.014 | 0.780 ± 0.006 | 0.936 ± 0.004 | 0.097 ± 0.005 | 3 |
| + class-balanced oversampling | 0.629 ± 0.023 | 0.528 ± 0.008 | 0.666 ± 0.029 | 0.923 ± 0.001 | 0.099 ± 0.026 | 3 |
| + GAN top-up (real + synthetic) | 0.447 ± 0.030 | 0.358 ± 0.069 | 0.578 ± 0.150 | 0.860 ± 0.043 | 0.112 ± 0.087 | 3 |
| + GAN mix (balanced, 50% synthetic per minority class) | 0.629 ± 0.032 | 0.521 ± 0.041 | 0.685 ± 0.043 | 0.919 ± 0.013 | 0.066 ± 0.020 | 3 |
| Synthetic only (1,000 GAN images per class; test on real) | 0.455 ± 0.013 | 0.371 ± 0.026 | 0.627 ± 0.051 | 0.855 ± 0.012 | 0.206 ± 0.053 | 3 |

## Per-class recall (lesion-level split, mean over seeds)

| Method | MEL (n=1113) | NV (n=6705) | BCC (n=514) | AKIEC (n=327) | BKL (n=1099) | DF (n=115) | VASC (n=142) |
|---|---:|---:|---:|---:|---:|---:|---:|
| erm | 0.33 | 0.88 | 0.43 | 0.28 | 0.32 | 0.13 | 0.68 |
| aug | 0.40 | 0.94 | 0.61 | 0.42 | 0.49 | 0.28 | 0.74 |
| weighted | 0.67 | 0.69 | 0.59 | 0.59 | 0.49 | 0.51 | 0.87 |
| focal | 0.39 | 0.94 | 0.59 | 0.44 | 0.48 | 0.30 | 0.69 |
| oversample | 0.68 | 0.70 | 0.66 | 0.57 | 0.50 | 0.46 | 0.83 |
| gan | 0.43 | 0.67 | 0.27 | 0.47 | 0.39 | 0.19 | 0.72 |
| gan_mix | 0.53 | 0.76 | 0.54 | 0.64 | 0.48 | 0.61 | 0.85 |
| synthetic_only | 0.37 | 0.76 | 0.26 | 0.43 | 0.35 | 0.19 | 0.82 |

## Split protocol: image-level vs lesion-level

| Method | Split | Test images whose lesion is also in train | Bal. acc | Accuracy | Macro-F1 |
|---|---|---:|---:|---:|---:|
| aug | image | 35.2% | 0.603 ± 0.010 | 0.798 ± 0.006 | 0.627 ± 0.002 |
| aug | lesion | 0.0% | 0.554 ± 0.013 | 0.785 ± 0.013 | 0.578 ± 0.015 |
| oversample | image | 35.2% | 0.712 ± 0.020 | 0.709 ± 0.009 | 0.623 ± 0.015 |
| oversample | lesion | 0.0% | 0.629 ± 0.023 | 0.666 ± 0.029 | 0.528 ± 0.008 |

## Does the GAN copy its training images?

Nearest-neighbour L2 distance (pixels in [0,1], 64×64×3) to the class's *training* images. If the generator memorised, synthetic images would sit much closer to training images than held-out real test images do.

| Class | Train images | Synthetic generated | Synthetic → train (median) | Real test → train (median) | Synthetic closer than 5th pct of real |
|---|---:|---:|---:|---:|---:|
| MEL | 779 | 3914 | 11.94 | 10.99 | 1.0% |
| BCC | 360 | 4333 | 8.75 | 8.24 | 0.7% |
| AKIEC | 229 | 4464 | 10.03 | 9.86 | 0.5% |
| BKL | 770 | 3923 | 9.07 | 8.69 | 0.3% |
| DF | 81 | 4612 | 9.32 | 8.64 | 2.9% |
| VASC | 99 | 4594 | 9.81 | 9.79 | 2.3% |
