"""Utility functions for the cable defect detection system."""

import cv2
import numpy as np
from pathlib import Path


def draw_detections(frame: np.ndarray, results, class_colors: dict = None) -> np.ndarray:
    """Draw bounding boxes and labels on a frame."""
    annotated = frame.copy()

    if class_colors is None:
        class_colors = {
            "sertissage_conforme": (0, 255, 0),
            "sertissage_non_conforme": (0, 0, 255),
            "denudage_conforme": (0, 255, 0),
            "denudage_non_conforme": (0, 0, 255),
        }

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_name = results[0].names[int(box.cls[0])]
        conf = float(box.conf[0])
        color = class_colors.get(cls_name, (255, 255, 0))

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{cls_name} {conf:.2f}"
        cv2.putText(annotated, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return annotated


def capture_frames(source: int = 0, count: int = 5, delay_ms: int = 200) -> list:
    """Capture multiple frames from camera with delay between each."""
    cap = cv2.VideoCapture(source)
    frames = []

    for _ in range(count):
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        cv2.waitKey(delay_ms)

    cap.release()
    return frames
