# Project Report — Industrial Defect Detection (Surface Crack Segmentation)

**Aim.** Automatically find and outline cracks on concrete, wall and road surfaces, and estimate the damaged
area, to support faster and safer infrastructure inspection.

**Data.** Crack Segmentation dataset: 4029 images with polygon labels for 5290 cracks, split
3717 / 200 / 112 into train / validation / test. Cracks are thin and small (median 1.1 % of image area).

**Method.** Instance segmentation with YOLO11n-seg pretrained on COCO, fine-tuned for 50 epochs at 640×640 with
mosaic, flip and colour augmentation (~18 minutes on an RTX 5080 Laptop GPU). The model outputs a box, class
score and pixel mask for each crack. The fraction of pixels covered by predicted masks is used as a simple
damage-severity indicator.

**Results (test set).** Box mAP@0.5 0.725, mask mAP@0.5 0.630, mask precision 0.716, mask recall 0.608,
1.8 ms per image. Mask mAP@0.5:0.95 is only 0.217 because thin cracks need near pixel-perfect masks to reach high
IoU; visually, predicted masks follow the cracks well but are often slightly thicker than the labels.

**Challenges.** Thin, branching shapes; background textures (stains, joints, shadows) that resemble cracks;
labels that are themselves approximate polygons; long training time compared to the other projects because of
the larger dataset. The original plan was road-pothole segmentation, but that public dataset is no longer
downloadable, so this closely related surface-defect task was chosen instead.

**Limitations and future work.** Severity is measured in % of image, not millimetres; use a calibrated camera
or a reference object to get real crack width. Higher input resolution (1024+) and a larger model (YOLO11s/m-seg)
should improve thin-crack masks; post-processing (skeletonisation) could measure crack length.

**Conclusion.** Instance segmentation localises cracks well enough for automatic screening and gives a
pixel-level damage estimate that a bounding-box detector cannot provide.
