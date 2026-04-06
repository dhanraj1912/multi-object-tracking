"""
main.py
=======
Pipeline entry point: video I/O + orchestration.

Usage
-----
    python main.py --input input.mp4 --output output.mp4

    # More detections in crowded scenes
    python main.py --input input.mp4 --output output.mp4 --conf 0.25

    # Show confidence scores on labels
    python main.py --input input.mp4 --output output.mp4 --show-conf

    # Tune DeepSORT directly
    python main.py --input input.mp4 --output output.mp4 \\
        --max-age 60 --n-init 3 --max-cosine-dist 0.35

    # See all options
    python main.py --help
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

from tracker import Tracker
from visualizer import Visualizer


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="YOLOv8 + DeepSORT multi-person tracking pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # I/O
    p.add_argument("--input",   default="input.mp4",  help="Input video path")
    p.add_argument("--output",  default="output.mp4", help="Output video path")

    # Detection
    p.add_argument("--model",   default="yolov8n.pt", help="YOLOv8 weights")
    p.add_argument("--conf",    type=float, default=0.30, help="YOLO confidence threshold")
    p.add_argument("--iou",     type=float, default=0.45, help="YOLO NMS IoU threshold")
    p.add_argument("--device",  default=None,
                   help="Inference device: cpu | 0 | 0,1  (default: auto)")

    # DeepSORT
    p.add_argument("--max-age",         type=int,   default=40,
                   help="Frames before a lost track is deleted")
    p.add_argument("--n-init",          type=int,   default=2,
                   help="Consecutive frames to confirm a new track")
    p.add_argument("--max-cosine-dist", type=float, default=0.4,
                   help="ReID appearance similarity gate (lower = stricter)")
    p.add_argument("--nn-budget",       type=int,   default=100,
                   help="Embedding gallery size per track")

    # Visualiser
    p.add_argument("--no-trails",  dest="trails",    action="store_false", default=True,
                   help="Disable motion trails")
    p.add_argument("--show-conf",  dest="show_conf", action="store_true",  default=False,
                   help="Show detection confidence on labels")

    return p.parse_args()


# ── Video helpers ──────────────────────────────────────────────────────────────

def open_video(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {path}")
        sys.exit(1)

    meta = {
        "width":  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps":    cap.get(cv2.CAP_PROP_FPS) or 25.0,
        "total":  int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    return cap, meta


def create_writer(path: str, meta: dict) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        path, fourcc, meta["fps"],
        (meta["width"], meta["height"]),
    )
    if not writer.isOpened():
        print(f"[ERROR] Cannot create output file: {path}")
        sys.exit(1)
    return writer


# ── Main pipeline ──────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:

    # ── Validate input ────────────────────────────────────────────────────
    if not Path(args.input).exists():
        print(f"[ERROR] Input file not found: {args.input}")
        sys.exit(1)

    cap, meta = open_video(args.input)
    writer    = create_writer(args.output, meta)

    print()
    print("=" * 60)
    print("  YOLOv8 + DeepSORT Tracking Pipeline")
    print("=" * 60)
    print(f"  Input   : {args.input}")
    print(f"  Output  : {args.output}")
    print(f"  Video   : {meta['width']}x{meta['height']} "
          f"@ {meta['fps']:.1f}fps  ({meta['total']} frames)")
    print(f"  Model   : {args.model}  conf={args.conf}  iou={args.iou}")
    print(f"  DeepSORT: max_age={args.max_age}  n_init={args.n_init}  "
          f"max_cosine_dist={args.max_cosine_dist}  nn_budget={args.nn_budget}")
    print("=" * 60)
    print()

    # ── Build pipeline ────────────────────────────────────────────────────
    tracker = Tracker(
        model_name=args.model,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        max_age=args.max_age,
        n_init=args.n_init,
        max_cosine_dist=args.max_cosine_dist,
        nn_budget=args.nn_budget,
    )

    visualizer = Visualizer(
        draw_trails=args.trails,
        show_confidence=args.show_conf,
    )

    # ── Process frames ────────────────────────────────────────────────────
    print("[INFO] Processing frames ...\n")
    frame_idx  = 0
    t_start    = time.time()
    total_time = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        t0 = time.time()

        # Detect + track
        tracks = tracker.update(frame)

        # Annotate
        annotated = visualizer.draw(frame, tracks, frame_idx=frame_idx)

        writer.write(annotated)

        elapsed = time.time() - t0
        total_time += elapsed
        frame_idx += 1

        # Progress every 50 frames
        if frame_idx % 50 == 0:
            pct     = frame_idx / max(meta["total"], 1) * 100
            avg_fps = frame_idx / (time.time() - t_start)
            print(f"  [{pct:5.1f}%]  frame {frame_idx:>5} / {meta['total']}"
                  f"  |  live tracks: {len(tracks):>3}"
                  f"  |  avg {avg_fps:.1f} fps")

    # ── Wrap up ───────────────────────────────────────────────────────────
    cap.release()
    writer.release()

    avg_ms  = (total_time / max(frame_idx, 1)) * 1000
    avg_fps = frame_idx / max(total_time, 1e-6)

    print()
    print("=" * 60)
    print("  Done!")
    print(f"  Saved to     : {args.output}")
    print(f"  Frames proc. : {frame_idx}")
    print(f"  Unique IDs   : {len(tracker.all_ids_seen)}")
    print(f"  Avg speed    : {avg_fps:.1f} fps  ({avg_ms:.1f} ms/frame)")
    print("=" * 60)
    print()


if __name__ == "__main__":
    run(parse_args())
