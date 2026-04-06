# Multi-Person Tracker - YOLOv8 + DeepSORT

A production-grade person tracking pipeline with stable, persistent IDs across frames — even through occlusion and re-appearance.

---
##Video link
https://drive.google.com/file/d/1deCuvX9JryZI-LTlxEo9i2syQZDjBzeA/view?usp=sharing
Approach

The pipeline is divided into two main stages:

##1. Detection

We use YOLOv8 for detecting people in each frame.

Fast and efficient
Performs well on real-world footage
Only the person class is considered

##2. Tracking

For tracking, we use DeepSORT.

Unlike simple trackers, DeepSORT uses:

Motion prediction (Kalman Filter)
Appearance features (Re-ID embeddings)

This allows the system to:

Keep track of individuals even after short occlusions
Reduce ID switching in crowded scenes

## Why DeepSORT over ByteTrack / SORT?

| Feature | SORT | ByteTrack | **DeepSORT** |
|---|---|---|---|
| Association method | IoU only | IoU (two-stage) | IoU + **appearance Re-ID** |
| Handles occlusion | ❌ Poor | ✅ Good | ✅✅ Best |
| Re-identifies after disappearance | ❌ No | ❌ No | ✅ Yes |
| ID switches in crowds | ❌ High | ⚠️ Medium | ✅ Low |
| Extra model required | No | No | Yes (lightweight MobileNet) |

**The key insight:** DeepSORT extracts a 128-dimensional appearance embedding (Re-ID feature) from each detected bounding box using a MobileNetV2 backbone. When matching detections to tracks, it blends:

- **Mahalanobis/IoU distance** → "is this roughly where you should be?"
- **Cosine distance on embeddings** → "do you *look* like the same person?"

This dual-cue matching means even after a 1–2 second occlusion, a person re-emerging from behind a pillar is re-identified as the same ID rather than getting a new one.

---

## Project Structure

```
.
├── main.py           # Entry point — CLI, video I/O, orchestration
├── detector.py       # YOLOv8 inference (person class only)
├── tracker.py        # DeepSORT multi-object tracking + Re-ID
├── visualizer.py     # Rendering: boxes, "ID: X" labels, trails, HUD
├── requirements.txt  # Python dependencies
└── README.md
```

---

## Setup

```bash
# 1. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

YOLOv8 weights (`yolov8n.pt`) download automatically on first run.

---

## Usage

```bash
# Basic run
python main.py --input input.mp4 --output output.mp4

# Lower confidence for crowded scenes
python main.py --input input.mp4 --output output.mp4 --conf 0.25

# Show confidence scores on labels
python main.py --input input.mp4 --output output.mp4 --show-conf

# Disable motion trails
python main.py --input input.mp4 --output output.mp4 --no-trails

# Tune DeepSORT parameters
python main.py --input input.mp4 --output output.mp4 \
    --max-age 60 --n-init 3 --max-cosine-dist 0.35

# Use a more accurate YOLO model
python main.py --input input.mp4 --output output.mp4 --model yolov8s.pt

# See all options
python main.py --help
```

---

## Parameter Tuning Guide

### `--conf` (default: 0.30)
YOLO detection confidence threshold.
- **Lower (0.20–0.25):** catches more people, including partially occluded or distant ones. More false positives.
- **Higher (0.40–0.50):** fewer false positives but misses hard cases.

### `--max-age` (default: 40)
How many consecutive frames a track can go unmatched before being deleted.
At 25fps, `max_age=40` = 1.6 seconds of tolerance.
- **Increase** for longer occlusions (crowds, doorways).
- **Decrease** if you see ghost tracks of people who have left.

### `--n-init` (default: 2)
Frames a new detection must be seen consecutively before getting a confirmed ID.
- **2:** fast confirmation, some flickering on false positives.
- **3–4:** slower confirmation, cleaner IDs.

### `--max-cosine-dist` (default: 0.4)
Appearance similarity gate. Two embeddings are considered "the same person" if cosine distance < this value.
- **Lower (0.2–0.3):** stricter Re-ID, fewer false merges, but may fail to re-identify across large appearance changes.
- **Higher (0.5–0.6):** more aggressive Re-ID, better recall on appearance changes, risk of merging different people.

---

## Output

- Annotated video with coloured bounding boxes per person
- `ID: N` pill labels above each box (unique colour per ID)
- Optional fading motion trail showing direction of travel
- HUD overlay: live person count + frame number
## Assumptions

- Only "person" class is tracked
- Camera motion is moderate (not extreme shake)
- Video resolution is sufficient for detection
- Subjects remain visible for at least a few frames for ID confirmation

## Limitations

- ID switches may occur in heavy occlusion or dense crowds
- Performance depends on video quality and lighting
- DeepSORT may struggle with identical-looking individuals
- Real-time performance may drop on CPU-only systems

- 
