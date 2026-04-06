"""
tracker.py
==========
Multi-object tracking using YOLOv8 detections + DeepSORT.

Why DeepSORT instead of SORT / ByteTrack?
------------------------------------------
SORT (Simple Online and Realtime Tracking) associates detections
across frames using only IoU overlap between predicted and observed
bounding boxes. This works well when targets are well-separated and
move slowly, but it fails in two common real-world scenarios:

  1. **Occlusion**: when a person walks behind an obstacle, their
     bounding box disappears for several frames. SORT loses the track
     and assigns a new ID when they re-emerge.

  2. **Crowding**: when two people cross paths and temporarily overlap,
     SORT often swaps their IDs because it cannot distinguish them —
     it only knows position, not appearance.

DeepSORT adds a second cue: **appearance embeddings (Re-ID features)**.
For each detection, a compact neural-network descriptor (128-d vector)
is extracted from the cropped bounding-box region. This vector encodes
what the person *looks like* — colour, texture, shape — independent of
their position. During association, the tracker blends two distances:

  • IoU / Mahalanobis distance  →  "are you in roughly the right place?"
  • Cosine distance on embeddings →  "do you look like the same person?"

The cosine distance is the key improvement. Even after a multi-second
occlusion, if the re-appearing crop looks similar to the stored gallery
of embeddings for track #5, the tracker can confidently re-assign ID 5
rather than starting ID 17.

Tuned parameters in this implementation
-----------------------------------------
max_age          = 40  frames   Tracks survive up to 40 consecutive
                                 frames without a detection match before
                                 being deleted. 40f ≈ 1.6s at 25 fps —
                                 enough to bridge most occlusions.

n_init           = 2   frames   A new track must be confirmed by at
                                 least 2 consecutive detections. This
                                 suppresses one-frame false positives
                                 (shadows, partial limbs) from getting IDs.

max_cosine_dist  = 0.4          Appearance similarity gate. Two embeddings
                                 are considered "the same person" if their
                                 cosine distance < 0.4. Lower = stricter
                                 (fewer re-id merges, safer). Higher = more
                                 aggressive re-id (may merge different people).

nn_budget        = 100          Gallery size: keep the last 100 appearance
                                 descriptors per track. A rolling gallery
                                 handles gradual appearance change (e.g.
                                 person turns around).

nms_max_overlap  = 0.85         NMS threshold inside DeepSORT's internal
                                 pre-processing step. High value so our
                                 upstream YOLO NMS is not double-applied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

from detector import Detector, RawDetection


@dataclass
class TrackedObject:
    """
    A confirmed, identity-stable tracked person.

    Attributes
    ----------
    track_id   : globally unique, persistent integer ID
    bbox       : [x1, y1, x2, y2] in pixels
    confidence : detection confidence that spawned this update
    centre     : (cx, cy) pixel centre of the bounding box
    age        : how many frames this track has been alive
    """
    track_id: int
    bbox: np.ndarray
    confidence: float
    centre: Tuple[int, int]
    age: int = 0


class Tracker:
    """
    Connects YOLOv8 detection → DeepSORT tracking.

    The detector and tracker are intentionally kept separate so either
    can be swapped independently (e.g. switch to YOLOv9 or StrongSORT).

    Parameters
    ----------
    model_name       : YOLOv8 weights filename
    conf             : YOLO detection confidence threshold
    iou              : YOLO NMS IoU threshold
    device           : inference device ('cpu', '0', etc.)
    max_age          : frames before a lost track is deleted
    n_init           : consecutive frames needed to confirm a track
    max_cosine_dist  : appearance similarity gate (lower = stricter)
    nn_budget        : max embeddings stored per track in the gallery
    """

    # ── DeepSORT defaults (tuned for pedestrian tracking at ~25 fps) ──
    _DEFAULT_MAX_AGE         = 40
    _DEFAULT_N_INIT          = 2
    _DEFAULT_MAX_COSINE_DIST = 0.4
    _DEFAULT_NN_BUDGET       = 100

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        conf: float = 0.30,
        iou: float = 0.45,
        device: Optional[str] = None,
        max_age: int = _DEFAULT_MAX_AGE,
        n_init: int = _DEFAULT_N_INIT,
        max_cosine_dist: float = _DEFAULT_MAX_COSINE_DIST,
        nn_budget: int = _DEFAULT_NN_BUDGET,
    ) -> None:

        # ── Detector ─────────────────────────────────────────────────────
        self.detector = Detector(
            model_name=model_name,
            conf=conf,
            iou=iou,
            device=device,
        )

        # ── DeepSORT tracker ─────────────────────────────────────────────
        # embedder="mobilenet" uses a lightweight MobileNetV2-based Re-ID
        # model that runs on CPU in real time. Switch to "clip_RN50" for
        # better appearance features at the cost of speed.
        print(f"[Tracker]  Initialising DeepSORT  "
              f"(max_age={max_age}, n_init={n_init}, "
              f"max_cosine_dist={max_cosine_dist}) ...")

        self._deepsort = DeepSort(
            max_age=max_age,
            n_init=n_init,
            nms_max_overlap=0.85,
            max_cosine_distance=max_cosine_dist,
            nn_budget=nn_budget,
            embedder="mobilenet",          # built-in Re-ID model
            half=False,                    # FP32 for MPS / CPU compatibility
            bgr=True,                      # our frames are OpenCV BGR
            embedder_gpu=False,            # set True if you have a CUDA GPU
        )

        # ── Bookkeeping ───────────────────────────────────────────────────
        self.all_ids_seen: Set[int] = set()

        print("[Tracker]  DeepSORT ready.")

    # ─────────────────────────────────────────────────────────────────────

    def update(self, frame: np.ndarray) -> List[TrackedObject]:
        """
        Detect people in `frame`, then update DeepSORT tracks.

        Steps
        -----
        1. YOLO detects all persons → RawDetection list
        2. Convert to DeepSORT input format
        3. DeepSORT matches detections to existing tracks using
           Kalman-filter predictions + appearance cosine distance
        4. Parse confirmed tracks → TrackedObject list

        Parameters
        ----------
        frame : np.ndarray
            BGR frame from cv2.VideoCapture.

        Returns
        -------
        List[TrackedObject]
            Only *confirmed* tracks (age >= n_init) are returned.
            Tentative tracks (first 1..n_init-1 frames) are suppressed
            to avoid flickering IDs on false-positive detections.
        """
        # Step 1: detect
        raw: List[RawDetection] = self.detector.detect(frame)

        # Step 2: convert  → [([x1,y1,x2,y2], conf, class_name), ...]
        ds_input = [d.to_deepsort_input() for d in raw]

        # Step 3: update DeepSORT — this is where Re-ID magic happens
        ds_tracks = self._deepsort.update_tracks(ds_input, frame=frame)

        # Step 4: collect confirmed tracks
        tracked: List[TrackedObject] = []

        for track in ds_tracks:
            if not track.is_confirmed():
                # Tentative tracks don't have stable IDs yet
                continue

            tid = int(track.track_id)
            ltrb = track.to_ltrb()                      # [x1, y1, x2, y2]
            x1, y1, x2, y2 = (int(v) for v in ltrb)
            bbox = np.array([x1, y1, x2, y2], dtype=int)

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Use stored detection conf if available, else default
            conf = track.det_conf if track.det_conf is not None else 0.0

            tracked.append(TrackedObject(
                track_id=tid,
                bbox=bbox,
                confidence=float(conf),
                centre=(cx, cy),
                age=track.age,
            ))
            self.all_ids_seen.add(tid)

        return tracked
