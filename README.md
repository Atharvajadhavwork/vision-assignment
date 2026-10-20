# Computer Vision Assignment — 5 Projects

**Student:** Atharva Avinash Jadhav

Five computer vision projects, each showing a different CV concept. Every folder is self-contained with its own
README, source code, requirements, results, evaluation and report.

| # | Project (from the assignment list) | CV concept | Dataset | Folder |
|---|---|---|---|---|
| 1 | Animal Detection for Smart Agriculture | Object detection | African Wildlife | [01_animal_detection](01_animal_detection) |
| 2 | Industrial Defect Detection System | Instance segmentation | Crack Segmentation | [02_defect_segmentation](02_defect_segmentation) |
| 3 | Medical Image Classification and Analysis | Image classification | PneumoniaMNIST (chest X-ray) | [03_medical_image_classification](03_medical_image_classification) |
| 4 | Multi-Object Tracking System | Detection + multi-object tracking | MOT16 | [04_multi_object_tracking](04_multi_object_tracking) |
| 5 | Vision-Based Human–Robot Interaction | Human pose estimation → robot motion | Own webcam recordings | [05_human_robot_interaction](05_human_robot_interaction) |

## Repository layout

```
<project>/
├── README.md          problem, objective, dataset, methodology, tools, results, conclusion
├── report.md          brief project report (incl. limitations & challenges)
├── requirements.txt
├── src/               source code
└── results/           metrics.json, plots, sample input/output images or video
```

## Quick start

```bash
conda create -n vision python=3.11 -y && conda activate vision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
cd 01_animal_detection
pip install -r requirements.txt
python src/main.py
```

Datasets are downloaded automatically by the scripts (except MOT16 — see its README).
Large files (datasets, training runs, weights) are not stored in the repo; they are recreated by running the code.

**Hardware used:** NVIDIA RTX 5080 Laptop GPU, Ubuntu, Python 3.11, PyTorch 2.11 (CUDA 12.8).
