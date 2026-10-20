"""Animal Detection for Smart Agriculture — YOLO11 object detection.

Steps: analyse dataset -> train -> evaluate on test split -> save sample predictions.
Run from this project folder:  python src/main.py
"""
import json
import shutil
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATA = "african-wildlife.yaml"  # Ultralytics downloads it automatically on first use
EPOCHS = 50


def analyse_dataset(info):
    """Count images and boxes per class in every split and plot the class balance."""
    names = info["names"]
    stats = {}
    for split in ("train", "val", "test"):
        label_dir = Path(info[split].replace("images", "labels"))
        counts = Counter()
        files = list(label_dir.glob("*.txt"))
        for f in files:
            for line in f.read_text().splitlines():
                counts[names[int(line.split()[0])]] += 1
        stats[split] = {"images": len(files), "boxes": dict(counts)}

    plt.figure(figsize=(6, 4))
    train = stats["train"]["boxes"]
    plt.bar(train.keys(), train.values(), color="tab:green")
    plt.title("Boxes per class (train split)")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(RESULTS / "class_distribution.png", dpi=120)
    plt.close()
    return stats


def main():
    RESULTS.mkdir(exist_ok=True)
    info = check_det_dataset(DATA)  # downloads if missing, returns paths + class names
    stats = analyse_dataset(info)

    # Preprocessing (resize to 640, normalise) and augmentation (mosaic, flips, HSV jitter)
    # are applied by the Ultralytics data loader during training.
    model = YOLO("yolo11n.pt")  # COCO-pretrained, fine-tuned here (transfer learning)
    model.train(data=DATA, epochs=EPOCHS, imgsz=640, batch=16, seed=0,
                project=str(ROOT / "runs"), name="train", exist_ok=True)

    best = YOLO(ROOT / "runs/train/weights/best.pt")
    m = best.val(data=DATA, split="test", project=str(ROOT / "runs"), name="test", exist_ok=True)
    metrics = {
        "precision": round(m.box.mp, 4),
        "recall": round(m.box.mr, 4),
        "mAP50": round(m.box.map50, 4),
        "mAP50-95": round(m.box.map, 4),
        "per_class_mAP50-95": {info["names"][c]: round(v, 4) for c, v in zip(m.box.ap_class_index, m.box.maps[m.box.ap_class_index])},
        "inference_ms_per_image": round(m.speed["inference"], 2),
        "dataset": stats,
    }
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2))

    for name in ("results.png", "confusion_matrix_normalized.png", "BoxPR_curve.png", "BoxF1_curve.png", "train_batch0.jpg"):
        for run in ("train", "test"):
            src = ROOT / "runs" / run / name
            if src.exists():
                shutil.copy(src, RESULTS / name)

    samples = RESULTS / "samples"
    samples.mkdir(exist_ok=True)
    test_imgs = sorted(Path(info["test"]).glob("*.jpg"))[::25][:8]
    for i, r in enumerate(best.predict(test_imgs, conf=0.4, verbose=False), start=1):
        r.save(filename=str(samples / f"pred_{i:02d}.jpg"))

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
