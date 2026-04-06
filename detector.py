"""
detector.py
===========
YOLOv8 inference module — person detection only.

Responsibilities
----------------
- Load a YOLOv8 model once at startup
- Run per-frame inference filtered to the 'person' class
- Return raw detections in the format DeepSORT expects:
      ([x1, y1, x2, y2], confidence, class_name)

No tracking logic lives here. This module is purely a detector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from ultralytics import YOLO

# COCO class index for 'person'
PERSON_CLASS_ID = 0
PERSON_CLASS_NAME = "person"


@dataclass
class RawDetection:
    """
    A single YOLO detection before any tracking.

    Attributes
    ----------
    bbox       : [x1, y1, x2, y2] in integer pixels
    confidence : YOLO confidence score in [0, 1]
    class_name : always 'person' in this pipeline
    """
    bbox: np.ndarray
    confidence: float
    class_name: str = PERSON_CLASS_NAME

    def to_deepsort_input(self) -> Tuple[List[int], float, str]:
        """
        Convert to the 3-tuple format expected by DeepSORT's update():
            ([x1, y1, x2, y2], confidence, class_name)
        """
        return (self.bbox.tolist(), self.confidence, self.class_name)


class Detector:
    """
    YOLOv8 person detector.

    Parameters
    ----------
    model_name : str
        YOLOv8 weights to use. 'yolov8n.pt' is fastest; use
        'yolov8s.pt' or 'yolov8m.pt' for better recall on small people.
        Weights are auto-downloaded by Ultralytics on first run.
    conf : float
        Minimum detection confidence. Lower values catch more people
        but increase false positives. 0.30 is a good general default.
    iou : float
        NMS IoU threshold. Increase for densely packed crowds to
        avoid suppressing nearby people; lower for isolated targets.
    device : str or None
        Inference device: 'cpu', '0' (GPU 0), '0,1' (multi-GPU).
        None lets Ultralytics choose automatically.
    """

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        conf: float = 0.30,
        iou: float = 0.45,
        device: Optional[str] = None,
    ) -> None:
        self.conf = conf
        self.iou = iou
        self.device = device

        print(f"[Detector] Loading {model_name} ...")
        self.model = YOLO(model_name)
        print("[Detector] Ready.")

    def detect(self, frame: np.ndarray) -> List[RawDetection]:
        """
        Run YOLOv8 inference on a single BGR frame.

        Parameters
        ----------
        frame : np.ndarray
            BGR image as returned by cv2.VideoCapture.read().

        Returns
        -------
        List[RawDetection]
            One entry per detected person, sorted by confidence (desc).
        """
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            classes=[PERSON_CLASS_ID],
            device=self.device,
            verbose=False,
        )

        detections: List[RawDetection] = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                detections.append(RawDetection(bbox=xyxy, confidence=conf))

        # Sort best-first so DeepSORT prioritises high-quality detections
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections
