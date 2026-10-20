"""Industrial Defect Detection System — surface crack segmentation with YOLO11-seg.

Steps: analyse dataset -> train -> evaluate on test split -> estimate defect area -> save sample predictions.
Run from this project folder:  python src/main.py
"""
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO
from ultralytics.data.utils import check_det_dataset

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DATA = "crack-seg.yaml"  # Ultralytics downloads it automatically on first use
EPOCHS = 50


def polygon_area(xy):
    """Shoelace formula; xy are normalised polygon points, so the result is a fraction of the image."""
    x, y = xy[:, 0], xy[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def analyse_dataset(info):
    """Count images/cracks and look at how big cracks are relative to the image."""
    stats, areas = {}, []
    for split in ("train", "val", "test"):
        files = list(Path(info[split].replace("images", "labels")).glob("*.txt"))
        n = 0
        for f in files:
            for line in f.read_text().splitlines():
                pts = np.array(line.split()[1:], dtype=float).reshape(-1, 2)
                areas.append(polygon_area(pts) * 100)
                n += 1
        stats[split] = {"images": len(files), "cracks": n}

    plt.figure(figsize=(6, 4))
    plt.hist(areas, bins=40, color="tab:orange")
    plt.title("Crack size (% of image area)")
    plt.xlabel("% of image")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(RESULTS / "crack_size_distribution.png", dpi=120)
    plt.close()
    stats["median_crack_area_pct"] = round(float(np.median(areas)), 2)
    return stats


def main():
    RESULTS.mkdir(exist_ok=True)
    info = check_det_dataset(DATA)
    stats = analyse_dataset(info)

    # Resizing, normalisation and augmentation (mosaic, flips, HSV jitter) happen in the data loader.
    model = YOLO("yolo11n-seg.pt")  # COCO-pretrained segmentation model, fine-tuned (transfer learning)
    model.train(data=DATA, epochs=EPOCHS, imgsz=640, batch=16, seed=0,
                project=str(ROOT / "runs"), name="train", exist_ok=True)

    best = YOLO(ROOT / "runs/train/weights/best.pt")
    m = best.val(data=DATA, split="test", project=str(ROOT / "runs"), name="test", exist_ok=True)
    metrics = {
        "box": {"precision": round(m.box.mp, 4), "recall": round(m.box.mr, 4),
                "mAP50": round(m.box.map50, 4), "mAP50-95": round(m.box.map, 4)},
        "mask": {"precision": round(m.seg.mp, 4), "recall": round(m.seg.mr, 4),
                 "mAP50": round(m.seg.map50, 4), "mAP50-95": round(m.seg.map, 4)},
        "inference_ms_per_image": round(m.speed["inference"], 2),
        "dataset": stats,
    }

    for name in ("results.png", "MaskPR_curve.png", "BoxPR_curve.png", "MaskF1_curve.png", "train_batch0.jpg"):
        for run in ("train", "test"):
            src = ROOT / "runs" / run / name
            if src.exists():
                shutil.copy(src, RESULTS / name)

    # Severity cue: the predicted mask tells us how much of the surface is damaged.
    samples = RESULTS / "samples"
    samples.mkdir(exist_ok=True)
    test_imgs = sorted(Path(info["test"]).glob("*.jpg"))[::15][:8]
    damage = {}
    for i, r in enumerate(best.predict(test_imgs, conf=0.4, verbose=False), start=1):
        name = f"pred_{i:02d}.jpg"
        r.save(filename=str(samples / name))
        cover = 0.0 if r.masks is None else r.masks.data.any(0).float().mean().item() * 100
        damage[name] = {"cracks": len(r.boxes), "damaged_area_pct": round(cover, 2)}
    metrics["sample_damage_estimate"] = damage

    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
