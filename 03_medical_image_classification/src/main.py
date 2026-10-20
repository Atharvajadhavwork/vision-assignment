"""Medical Image Classification — pneumonia vs normal chest X-rays with YOLO11-cls.

Steps: download + convert dataset -> analyse -> train -> evaluate on test split -> save samples.
Run from this project folder:  python src/main.py
"""
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from medmnist import PneumoniaMNIST
from PIL import Image
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RAW = ROOT / "datasets/raw"
DATA = ROOT / "datasets/pneumonia"
CLASSES = ["normal", "pneumonia"]
SIZE = 128  # MedMNIST+ resolution (28, 64, 128 or 224)
EPOCHS = 30


def prepare_dataset():
    """Download PneumoniaMNIST and write it as folders: <split>/<class>/<id>.png (what YOLO-cls expects)."""
    RAW.mkdir(parents=True, exist_ok=True)
    stats = {}
    for split in ("train", "val", "test"):
        ds = PneumoniaMNIST(split=split, download=True, size=SIZE, root=str(RAW))
        labels = ds.labels.ravel()
        stats[split] = {c: int((labels == i).sum()) for i, c in enumerate(CLASSES)}
        if (DATA / split).exists():
            continue
        for c in CLASSES:
            (DATA / split / c).mkdir(parents=True, exist_ok=True)
        for i, (img, y) in enumerate(zip(ds.imgs, labels)):
            Image.fromarray(img).save(DATA / split / CLASSES[y] / f"{i:05d}.png")
    return stats


def analyse_dataset(stats):
    """Plot class balance per split and one example image from each class."""
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.5))
    x = np.arange(3)
    for k, c in enumerate(CLASSES):
        ax[0].bar(x + k * 0.4, [stats[s][c] for s in ("train", "val", "test")], 0.4, label=c)
    ax[0].set_xticks(x + 0.2, ["train", "val", "test"])
    ax[0].set_title("Images per class")
    ax[0].legend()
    for k, c in enumerate(CLASSES):
        img = Image.open(next((DATA / "train" / c).glob("*.png")))
        ax[k + 1].imshow(img, cmap="gray")
        ax[k + 1].set_title(f"example: {c}")
        ax[k + 1].axis("off")
    plt.tight_layout()
    plt.savefig(RESULTS / "dataset_overview.png", dpi=120)
    plt.close()


def main():
    RESULTS.mkdir(exist_ok=True)
    stats = prepare_dataset()
    analyse_dataset(stats)

    # Grayscale X-rays are loaded as 3-channel images, resized and normalised by the data loader;
    # flips / small HSV and scale changes are used as augmentation.
    model = YOLO("yolo11n-cls.pt")  # ImageNet-pretrained, fine-tuned (transfer learning)
    model.train(data=str(DATA), epochs=EPOCHS, imgsz=SIZE, batch=64, seed=0,
                project=str(ROOT / "runs"), name="train", exist_ok=True)
    shutil.copy(ROOT / "runs/train/results.png", RESULTS / "training_curves.png")

    # Evaluate on the held-out test split with clinical metrics (sensitivity matters most).
    best = YOLO(ROOT / "runs/train/weights/best.pt")
    paths, y_true = [], []
    for i, c in enumerate(CLASSES):
        files = sorted((DATA / "test" / c).glob("*.png"))
        paths += files
        y_true += [i] * len(files)
    y_true = np.array(y_true)
    probs = np.array([r.probs.data.cpu().numpy() for r in best.predict(paths, imgsz=SIZE, batch=64, verbose=False, stream=True)])
    y_pred = probs.argmax(1)

    report = classification_report(y_true, y_pred, target_names=CLASSES, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "accuracy": round(report["accuracy"], 4),
        "sensitivity_recall_pneumonia": round(tp / (tp + fn), 4),
        "specificity": round(tn / (tn + fp), 4),
        "precision_pneumonia": round(report["pneumonia"]["precision"], 4),
        "f1_pneumonia": round(report["pneumonia"]["f1-score"], 4),
        "roc_auc": round(roc_auc_score(y_true, probs[:, 1]), 4),
        "confusion_matrix": cm.tolist(),
        "dataset": stats,
    }
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2))

    ConfusionMatrixDisplay(cm, display_labels=CLASSES).plot(cmap="Blues")
    plt.title("Test confusion matrix")
    plt.savefig(RESULTS / "confusion_matrix.png", dpi=120, bbox_inches="tight")
    plt.close()
    fpr, tpr, _ = roc_curve(y_true, probs[:, 1])
    plt.plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate (sensitivity)")
    plt.title("ROC curve (test)")
    plt.legend()
    plt.savefig(RESULTS / "roc_curve.png", dpi=120, bbox_inches="tight")
    plt.close()

    # Sample grid: 4 correct and 4 wrong predictions, so both successes and failures are visible.
    right = np.where(y_pred == y_true)[0][::97][:4]
    wrong = np.where(y_pred != y_true)[0][:4]
    fig, ax = plt.subplots(2, 4, figsize=(10, 5.5))
    for a, idx in zip(ax.ravel(), np.concatenate([right, wrong])):
        a.imshow(Image.open(paths[idx]), cmap="gray")
        ok = "OK" if y_pred[idx] == y_true[idx] else "WRONG"
        a.set_title(f"{ok}\ntrue={CLASSES[y_true[idx]]}\npred={CLASSES[y_pred[idx]]} ({probs[idx].max():.2f})", fontsize=8)
        a.axis("off")
    plt.tight_layout()
    plt.savefig(RESULTS / "sample_predictions.png", dpi=120)
    plt.close()

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
