# Project Report — Animal Detection for Smart Agriculture

**Aim.** Automatically detect and identify large animals (buffalo, elephant, rhino, zebra) in camera images so
that farms can be alerted to animals entering fields.

**Data.** African Wildlife dataset (1504 images, 4 classes, YOLO box labels), split 1052 / 225 / 227 for
train / validation / test. Classes are roughly balanced (399–575 boxes per class in training).

**Method.** Transfer learning with YOLO11n pretrained on COCO. Images are resized to 640×640 and normalised;
mosaic, flips and colour jitter augment the small training set. The model was trained for 50 epochs
(~8 minutes on an RTX 5080 Laptop GPU) and the checkpoint with the best validation mAP was kept.

**Results (test set).** Precision 0.963, recall 0.935, mAP@0.5 0.971, mAP@0.5:0.95 0.828 at 2.5 ms per image.
Rhino is detected best (0.91 mAP@0.5:0.95); elephant (0.77) and zebra (0.78) are hardest because they appear in
overlapping groups. The confusion matrix shows almost no confusion between species — most errors are missed
or loosely-boxed animals, not wrong labels.

**Challenges.** Overlapping animals in herds, varying scale (very near vs. far away animals) and a small
dataset. Pretraining and mosaic augmentation were essential to reach high accuracy with only ~1000 images.

**Limitations and future work.** Only 4 savanna species and daytime images. For a real farm: collect local
animals (cattle, boar, deer, dogs), add night/infrared images, test on real farm-camera footage, and connect
detections to an alert system (SMS/siren).

**Conclusion.** Fine-tuned YOLO11n gives accurate (0.97 mAP@0.5) and very fast animal detection, making
real-time monitoring of fields practical on low-cost hardware.
