# Technical Report: Multi-Object Detection and Persistent ID Tracking

## 1. Introduction

This project implements a computer vision pipeline for **multi-object detection and persistent identity tracking** in public sports/event video footage. The system detects multiple individuals and assigns **consistent, unique IDs** that remain stable across frames, even under real-world challenges such as occlusion, motion, and crowd density.

---

## 2. Model / Detector Used

For object detection, we use **YOLOv8 (You Only Look Once version 8)**.

* **Model**: YOLOv8n (nano version)
* **Reasons for selection**:

  * Fast inference suitable for near real-time processing
  * Good accuracy in dynamic scenes
  * Built-in Non-Maximum Suppression (NMS)
  * Easy integration via Ultralytics API

The detector is configured to track only the **person class**, as required for this task.

---

## 3. Tracking Algorithm Used

We use **DeepSORT (Deep Simple Online and Realtime Tracking)** for multi-object tracking.

### Core Components:

* **Kalman Filter** → predicts object motion
* **Hungarian Algorithm** → assigns detections to tracks
* **Appearance Embeddings (Re-ID)** → identifies objects based on visual features

A lightweight **MobileNet-based Re-ID model** extracts a 128-dimensional feature vector for each detected subject.

---

## 4. Why YOLOv8 + DeepSORT?

This combination was selected because:

* YOLOv8 provides **fast and accurate detection**
* DeepSORT provides **robust identity tracking**

Unlike simpler trackers (e.g., SORT, ByteTrack), DeepSORT uses **appearance-based matching**, which enables:

* Re-identification after occlusion
* Reduced ID switching in crowded scenes
* Improved tracking of similar-looking individuals

---

## 5. How ID Consistency is Maintained

ID consistency is ensured through:

### 1. Motion-based tracking

* Kalman filter predicts next object positions
* Maintains continuity during missed detections

### 2. Appearance-based matching

* Each detection is encoded into a feature vector
* Cosine similarity compares new detections with past identities
* Helps maintain identity even after occlusion

### 3. Parameter tuning

* `max_age`: allows temporary disappearance
* `n_init`: filters out false positives
* `max_cosine_dist`: controls appearance similarity threshold

---

## 6. Challenges Faced

* **Occlusion**: subjects overlap or disappear
* **Similar appearance**: difficult to distinguish individuals
* **Motion blur**: fast movement reduces detection quality
* **Camera motion**: zoom and pan affect stability

---

## 7. Failure Cases Observed

* ID switching in very dense crowds
* Loss of tracking after long occlusion
* Reduced accuracy in low-resolution or poorly lit videos

---

## 8. Possible Improvements

* Use stronger Re-ID models (e.g., ResNet, CLIP-based embeddings)
* Apply temporal smoothing for stability
* Upgrade to larger YOLO models (yolov8m/l)
* Add multi-camera tracking
* Implement behavior analysis (e.g., clustering, speed estimation)

---

## 9. Conclusion

The system successfully demonstrates **multi-object detection and persistent identity tracking** in real-world scenarios. By combining YOLOv8 with DeepSORT, it achieves reliable tracking performance, even under challenging conditions such as occlusion and crowd interactions.

---
