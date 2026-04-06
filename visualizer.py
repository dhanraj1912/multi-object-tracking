"""
visualizer.py
=============
All rendering logic for the DeepSORT tracking pipeline.

Design principles
-----------------
- One stable colour per track ID (golden-ratio hue spacing)
- Clean "ID: X" label with auto-contrast text on a filled pill
- Optional fading motion trail showing recent path
- Minimal HUD showing live person count and frame number
- Zero business logic — pure drawing code only
"""

from __future__ import annotations

import colorsys
from collections import defaultdict, deque
from typing import Dict, List, Tuple

import cv2
import numpy as np

from tracker import TrackedObject

# ── Visual constants ──────────────────────────────────────────────────────────
TRAIL_LENGTH  = 50          # historical centre points kept per track
BOX_THICKNESS = 2           # bounding box line width in pixels
LABEL_PAD_X   = 8           # horizontal padding inside the ID label pill
LABEL_PAD_Y   = 5           # vertical padding inside the ID label pill
FONT          = cv2.FONT_HERSHEY_DUPLEX
FONT_SCALE    = 0.58
FONT_THICK    = 1


class Visualizer:
    """
    Renders annotations onto video frames.

    Parameters
    ----------
    draw_trails : bool
        If True, draw a fading trail behind each tracked person.
    show_confidence : bool
        If True, append the detection confidence to the ID label.
    """

    def __init__(
        self,
        draw_trails: bool = True,
        show_confidence: bool = False,
    ) -> None:
        self.draw_trails      = draw_trails
        self.show_confidence  = show_confidence

        self._colours: Dict[int, Tuple[int, int, int]] = {}
        self._trails: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=TRAIL_LENGTH)
        )

    def draw(
        self,
        frame: np.ndarray,
        tracks: List[TrackedObject],
        frame_idx: int = 0,
    ) -> np.ndarray:
        """
        Annotate a single frame and return the result.

        Rendering order (back → front):
            trails → bounding boxes → ID labels → HUD overlay

        Parameters
        ----------
        frame     : BGR image from OpenCV
        tracks    : confirmed tracked objects for this frame
        frame_idx : frame counter for the HUD display

        Returns
        -------
        np.ndarray  annotated BGR frame
        """
        canvas = frame.copy()

        # Update motion history
        for t in tracks:
            self._trails[t.track_id].append(t.centre)

        # ── Layer 1: motion trails (drawn first — behind boxes) ───────────
        if self.draw_trails:
            for t in tracks:
                self._draw_trail(canvas, t.track_id)

        # ── Layer 2: bounding boxes + ID labels ───────────────────────────
        for t in tracks:
            colour = self._colour_for(t.track_id)
            self._draw_box(canvas, t.bbox, colour)
            self._draw_label(canvas, t, colour)

        # ── Layer 3: HUD ──────────────────────────────────────────────────
        self._draw_hud(canvas, count=len(tracks), frame_idx=frame_idx)

        return canvas

    # ── Drawing primitives ────────────────────────────────────────────────

    def _draw_box(
        self,
        canvas: np.ndarray,
        bbox: np.ndarray,
        colour: Tuple[int, int, int],
    ) -> None:
        """Draw a clean, anti-aliased bounding box."""
        x1, y1, x2, y2 = bbox
        cv2.rectangle(
            canvas, (x1, y1), (x2, y2),
            colour, BOX_THICKNESS, cv2.LINE_AA,
        )

    def _draw_label(
        self,
        canvas: np.ndarray,
        track: TrackedObject,
        colour: Tuple[int, int, int],
    ) -> None:
        """
        Draw a filled pill label above the bounding box.

        Format: "ID: <N>"  or  "ID: <N>  0.87" when show_confidence=True.
        The label is auto-clipped to stay inside the frame top edge.
        """
        if self.show_confidence:
            label = f"ID: {track.track_id}  {track.confidence:.2f}"
        else:
            label = f"ID: {track.track_id}"

        (tw, th), baseline = cv2.getTextSize(label, FONT, FONT_SCALE, FONT_THICK)

        x1, y1 = int(track.bbox[0]), int(track.bbox[1])

        # Pill rectangle (clipped to frame)
        pill_x1 = x1
        pill_y1 = max(0, y1 - th - 2 * LABEL_PAD_Y - baseline)
        pill_x2 = x1 + tw + 2 * LABEL_PAD_X
        pill_y2 = y1

        cv2.rectangle(canvas, (pill_x1, pill_y1), (pill_x2, pill_y2),
                      colour, -1, cv2.LINE_AA)

        text_x = pill_x1 + LABEL_PAD_X
        text_y = pill_y2 - baseline - (LABEL_PAD_Y // 2)

        text_colour = _contrasting_colour(colour)
        cv2.putText(
            canvas, label, (text_x, text_y),
            FONT, FONT_SCALE, text_colour, FONT_THICK, cv2.LINE_AA,
        )

    def _draw_trail(self, canvas: np.ndarray, track_id: int) -> None:
        """
        Draw a fading polyline of recent centre points.

        Older segments are thinner and dimmer, creating a comet-tail effect
        that shows direction of travel without cluttering the frame.
        """
        pts = list(self._trails[track_id])
        if len(pts) < 2:
            return

        colour = self._colour_for(track_id)
        n = len(pts)

        for i in range(1, n):
            alpha = i / n                              # 0 = old → 1 = recent
            faded = tuple(int(c * alpha) for c in colour)
            thick = max(1, int(3 * alpha))
            cv2.line(canvas, pts[i - 1], pts[i], faded, thick, cv2.LINE_AA)

    def _draw_hud(
        self,
        canvas: np.ndarray,
        count: int,
        frame_idx: int,
    ) -> None:
        """
        Top-right semi-transparent overlay showing:
          • Live person count
          • Current frame number
        """
        lines = [
            f"Tracking: {count} people",
            f"Frame: {frame_idx}",
        ]

        font, scale, thick = cv2.FONT_HERSHEY_DUPLEX, 0.48, 1
        line_h = 22
        pad = 10

        # Measure widest line
        max_w = max(cv2.getTextSize(l, font, scale, thick)[0][0] for l in lines)

        h, w = canvas.shape[:2]
        box_x1 = w - max_w - pad * 3
        box_y1 = pad
        box_x2 = w - pad
        box_y2 = pad + len(lines) * line_h + pad

        # Semi-transparent dark background
        overlay = canvas.copy()
        cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2),
                      (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.55, canvas, 0.45, 0, canvas)

        for i, line in enumerate(lines):
            y = box_y1 + pad + (i + 1) * line_h - 4
            cv2.putText(canvas, line, (box_x1 + pad, y),
                        font, scale, (230, 230, 230), thick, cv2.LINE_AA)

    # ── Colour utilities ──────────────────────────────────────────────────

    def _colour_for(self, track_id: int) -> Tuple[int, int, int]:
        """
        Return a stable, visually distinct BGR colour for a given track ID.

        Uses golden-ratio (φ) hue spacing so consecutive IDs are never
        adjacent on the colour wheel — maximising visual separation even
        with many simultaneous tracks.
        """
        if track_id not in self._colours:
            hue = (track_id * 0.618033988749895) % 1.0   # golden ratio
            r, g, b = colorsys.hsv_to_rgb(hue, 0.82, 0.95)
            self._colours[track_id] = (
                int(b * 255), int(g * 255), int(r * 255),  # BGR
            )
        return self._colours[track_id]


# ── Module-level helper ───────────────────────────────────────────────────────

def _contrasting_colour(bg: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """
    Return near-black or near-white based on perceived luminance of `bg`.
    Uses the ITU-R BT.601 luma formula (in BGR order).
    """
    b, g, r = bg
    luma = 0.299 * r + 0.587 * g + 0.114 * b
    return (15, 15, 15) if luma > 140 else (240, 240, 240)
