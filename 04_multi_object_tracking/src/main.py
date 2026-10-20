"""Multi-Object Tracking System — YOLO11 pedestrian detection + ByteTrack / BoT-SORT tracking on MOT16.

Steps: analyse dataset -> track every train sequence with both trackers -> score with CLEAR-MOT / IDF1
-> save comparison chart, sample frames and a short demo video.
Run from this project folder:  python src/main.py
(MOT16 must be unzipped to datasets/MOT16 — see README for the download command.)
"""
import json
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import motmetrics as mm
import numpy as np
import pandas as pd
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MOT = ROOT / "datasets/MOT16/train"
TRACKERS = {"ByteTrack": "bytetrack.yaml", "BoT-SORT": "botsort.yaml"}
DETECTOR = "yolo11m.pt"  # COCO-pretrained; class 0 = person
DEMO_SEQ, DEMO_FRAMES = "MOT16-09", 200


def load_gt(seq):
    """MOT16 gt.txt: frame, id, x, y, w, h, consider_flag, class, visibility. Keep pedestrians (class 1) only."""
    gt = pd.read_csv(seq / "gt/gt.txt", header=None,
                     names=["frame", "id", "x", "y", "w", "h", "flag", "cls", "vis"])
    return gt[(gt.flag == 1) & (gt.cls == 1)]


def analyse_dataset(seqs):
    rows = []
    for seq in seqs:
        gt = load_gt(seq)
        rows.append({"sequence": seq.name, "frames": len(list((seq / "img1").glob("*.jpg"))),
                     "pedestrian_ids": gt.id.nunique(), "boxes": len(gt),
                     "people_per_frame": round(gt.groupby("frame").size().mean(), 1)})
    df = pd.DataFrame(rows)
    df.plot.bar(x="sequence", y="people_per_frame", legend=False, color="tab:blue", figsize=(7, 3.5),
                title="Average pedestrians per frame (crowd density)")
    plt.tight_layout()
    plt.savefig(RESULTS / "dataset_density.png", dpi=120)
    plt.close()
    return df


def track_sequence(seq, tracker, demo_writer=None):
    """Run detection + tracking frame by frame and score it against ground truth."""
    gt = load_gt(seq)
    acc = mm.MOTAccumulator(auto_id=True)
    model = YOLO(DETECTOR)  # new model per sequence = fresh tracker state
    start = time.time()
    frames = sorted((seq / "img1").glob("*.jpg"))
    for f, r in enumerate(model.track(source=str(seq / "img1"), tracker=tracker, classes=[0], conf=0.3,
                                      imgsz=1280, persist=True, stream=True, verbose=False), start=1):
        g = gt[gt.frame == f]
        if r.boxes.id is not None:
            xyxy = r.boxes.xyxy.cpu().numpy()
            hyp = np.c_[xyxy[:, :2], xyxy[:, 2:] - xyxy[:, :2]]  # -> x, y, w, h like the ground truth
            ids = r.boxes.id.int().cpu().tolist()
        else:
            hyp, ids = np.empty((0, 4)), []
        acc.update(g.id.tolist(), ids, mm.distances.iou_matrix(g[["x", "y", "w", "h"]].values, hyp, max_iou=0.5))
        if demo_writer is not None and f <= DEMO_FRAMES:
            demo_writer(f, r.plot(line_width=2, font_size=12))
    fps = len(frames) / (time.time() - start)
    return acc, fps


def main():
    RESULTS.mkdir(exist_ok=True)
    seqs = sorted(p for p in MOT.iterdir() if p.is_dir())
    dataset = analyse_dataset(seqs)

    mh = mm.metrics.create()
    summaries, speed = {}, {}
    for name, cfg in TRACKERS.items():
        accs, fps = [], []
        for seq in seqs:
            writer = None
            if name == "ByteTrack" and seq.name == DEMO_SEQ:
                video = cv2.VideoWriter(str(RESULTS / "demo_tracking.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 15, (960, 540))

                def writer(f, img):
                    small = cv2.resize(img, (960, 540))
                    video.write(small)
                    if f in (20, 100, 180):
                        cv2.imwrite(str(RESULTS / f"sample_frame_{f:03d}.jpg"), small)
            acc, s = track_sequence(seq, cfg, writer)
            if writer is not None:
                video.release()
            accs.append(acc)
            fps.append(s)
            print(f"{name} {seq.name}: {s:.1f} FPS")
        summaries[name] = mh.compute_many(accs, metrics=mm.metrics.motchallenge_metrics,
                                          names=[s.name for s in seqs], generate_overall=True)
        speed[name] = round(float(np.mean(fps)), 1)

    overall = {}
    for name, summ in summaries.items():
        o = summ.loc["OVERALL"]
        overall[name] = {"MOTA": round(o.mota, 4), "MOTP_iou_dist": round(o.motp, 4), "IDF1": round(o.idf1, 4),
                         "ID_switches": int(o.num_switches), "false_positives": int(o.num_false_positives),
                         "misses": int(o.num_misses), "mostly_tracked": int(o.mostly_tracked),
                         "mostly_lost": int(o.mostly_lost), "avg_fps": speed[name]}
        text = mm.io.render_summary(summ, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names)
        (RESULTS / f"per_sequence_{name}.txt").write_text(text)
    metrics = {"detector": DETECTOR, "overall": overall, "dataset": dataset.to_dict("records")}
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2))

    fig, ax = plt.subplots(1, 3, figsize=(11, 3.5))
    for a, key in zip(ax, ["MOTA", "IDF1", "ID_switches"]):
        a.bar(overall.keys(), [overall[t][key] for t in overall], color=["tab:blue", "tab:orange"])
        a.set_title(key)
    plt.suptitle("Tracker comparison on MOT16 train (same YOLO11m detections)")
    plt.tight_layout()
    plt.savefig(RESULTS / "tracker_comparison.png", dpi=120)
    plt.close()

    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
