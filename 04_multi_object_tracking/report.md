# Project Report — Multi-Object Tracking System

**Aim.** Track every pedestrian in a video with a consistent ID and compare two state-of-the-art trackers.

**Data.** MOT16 train set: 7 sequences, 5316 frames, 517 identities, ranging from sparse street scenes
(8 people per frame) to dense crowds (45 people per frame), with both static and moving cameras.

**Method.** Tracking-by-detection. A COCO-pretrained YOLO11m detects people in each frame (1280 px input,
confidence 0.3). ByteTrack and BoT-SORT then predict each track's next position with a Kalman filter and match
new detections by IoU; BoT-SORT also compensates for camera motion. Outputs were scored against ground truth
using py-motmetrics (IoU ≥ 0.5).

**Results.** BoT-SORT: MOTA 42.2 %, IDF1 55.2 %, 557 ID switches, 18 FPS.
ByteTrack: MOTA 40.9 %, IDF1 52.3 %, 799 ID switches, 33 FPS.
BoT-SORT keeps identities better, especially on moving-camera sequences; ByteTrack is nearly twice as fast.

**Challenges.**
- Crowds and occlusion: people walking behind each other are missed or swap IDs.
- Small distant pedestrians — handled partly by using a larger 1280 px input.
- Tooling: the PyPI release of `py-motmetrics` crashes with NumPy 2 (`np.asfarray` was removed), so the fixed
  GitHub version is installed (see `requirements.txt`).
- Each sequence needs a fresh tracker state, otherwise IDs leak from one video into the next.

**Limitations and future work.** Missed detections are the largest error source, so fine-tuning the detector
on pedestrian datasets is the next step; adding an appearance (Re-ID) model would further reduce ID switches.
The evaluation is a simplified version of the official kit, and FPS was measured on a shared GPU.

**Conclusion.** A tracking pipeline built from a pretrained detector and an off-the-shelf tracker achieves
reasonable zero-shot results on MOT16. BoT-SORT is preferred for moving cameras such as robots; ByteTrack for
speed-critical fixed cameras.
