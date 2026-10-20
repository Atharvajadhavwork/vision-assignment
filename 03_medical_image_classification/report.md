# Project Report — Pneumonia Detection from Chest X-rays

**Aim.** Classify paediatric chest X-rays as normal or pneumonia to help prioritise radiologist review.

**Data.** PneumoniaMNIST (5856 X-rays from Kermany et al. 2018) at 128×128 resolution, with the official
train / val / test split (4708 / 524 / 624). Training data is imbalanced: 3494 pneumonia vs 1214 normal.

**Method.** The images were converted into a folder-per-class layout and used to fine-tune an
ImageNet-pretrained YOLO11n-cls network for 30 epochs (under 1 minute on an RTX 5080). Images are normalised and
augmented with flips, scaling, brightness changes and random erasing. The best validation checkpoint was
tested on the held-out test set.

**Results.** Accuracy 94.4 %, sensitivity 99.5 % (388 / 390 pneumonia cases found), specificity 85.9 %,
F1 0.957 and ROC-AUC 0.995.

**Analysis.** Errors are almost entirely one-sided: 33 normal images predicted as pneumonia vs only 2 missed
pneumonia cases. The imbalanced training data makes "pneumonia" the model's default guess, and some false positives
are predicted with high confidence (see `results/sample_predictions.png`), so they are not just borderline cases. Because the ROC-AUC is very high,
raising the decision threshold would trade a little sensitivity for much better specificity.

**Challenges.** Class imbalance; low resolution; medical metrics (sensitivity vs specificity) matter more than
plain accuracy; converting the `.npz` dataset to the image-folder format expected by the framework.

**Limitations and future work.** Single-hospital paediatric data; low resolution; no explanation of *why* a
prediction was made. Next steps: class-weighted loss or threshold tuning, higher-resolution (224) images,
Grad-CAM heat-maps to show which lung regions drive the decision, and testing on external datasets.

**Conclusion.** A small CNN classifier achieves screening-grade sensitivity (99.5 %) on unseen X-rays; reducing
false positives is the main improvement needed.
