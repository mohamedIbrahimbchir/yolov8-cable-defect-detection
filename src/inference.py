"""
Real-time Inference Script for Raspberry Pi 5
Cable Crimping & Stripping Defect Detection

Implements multi-image voting logic with FC1 safety rule:
    - 5 consecutive images are captured
    - All 5 must agree for a "conforme" decision
    - No detection = NOK (safety-first approach)

Usage:
    python src/inference.py --model models/best_ncnn_model --source 0
"""

import argparse
import time
from collections import Counter

import cv2
from ultralytics import YOLO


# FC1 Safety Rule: absence of detection != conformity
CONFORMITY_CLASSES = {"sertissage_conforme", "denudage_conforme"}
VOTE_COUNT = 5
CONFIDENCE_THRESHOLD = 0.5


def classify_frame(model: YOLO, frame, conf: float = CONFIDENCE_THRESHOLD) -> str:
    """
    Run inference on a single frame and return the dominant class.

    Returns:
        Class name, or "NO_DETECTION" if nothing is found.
    """
    results = model.predict(frame, conf=conf, verbose=False)

    if len(results[0].boxes) == 0:
        return "NO_DETECTION"

    # Get the highest-confidence detection
    boxes = results[0].boxes
    best_idx = boxes.conf.argmax().item()
    class_id = int(boxes.cls[best_idx].item())
    return model.names[class_id]


def multi_image_vote(predictions: list[str]) -> tuple[str, bool]:
    """
    Apply multi-image voting logic.

    Args:
        predictions: List of class predictions from consecutive frames.

    Returns:
        (decision, is_conforme): Final decision string and conformity flag.
    """
    counter = Counter(predictions)
    most_common, count = counter.most_common(1)[0]

    # FC1 Safety Rule: no detection = NOK
    if "NO_DETECTION" in predictions:
        return "NOK - No detection (safety rule)", False

    # All must agree for conformity
    if count == len(predictions) and most_common in CONFORMITY_CLASSES:
        return f"OK - {most_common}", True

    # Any non-conforme detection = NOK
    return f"NOK - {most_common} ({count}/{len(predictions)})", False


def run_inference(model_path: str, source: int = 0) -> None:
    """Main inference loop with multi-image voting."""
    model = YOLO(model_path)
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Error: Cannot open camera source {source}")
        return

    print(f"Starting inference (voting: {VOTE_COUNT} frames)...")
    print("Press 'q' to quit\n")

    predictions = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        prediction = classify_frame(model, frame)
        predictions.append(prediction)

        # Display current prediction on frame
        color = (0, 255, 0) if prediction in CONFORMITY_CLASSES else (0, 0, 255)
        cv2.putText(frame, prediction, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Vote every N frames
        if len(predictions) >= VOTE_COUNT:
            decision, is_ok = multi_image_vote(predictions[-VOTE_COUNT:])
            status_color = (0, 255, 0) if is_ok else (0, 0, 255)
            cv2.putText(frame, decision, (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            print(f"Decision: {decision}")
            predictions.clear()

        cv2.imshow("Cable Defect Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cable defect detection inference")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to NCNN model directory or .pt file")
    parser.add_argument("--source", type=int, default=0,
                        help="Camera source index (default: 0)")
    parser.add_argument("--conf", type=float, default=CONFIDENCE_THRESHOLD,
                        help=f"Confidence threshold (default: {CONFIDENCE_THRESHOLD})")
    args = parser.parse_args()

    CONFIDENCE_THRESHOLD = args.conf
    run_inference(args.model, args.source)
