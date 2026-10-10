"""
YOLOv8 Training Script for Cable Defect Detection
LEONI Wiring Systems Tunisia — Summer Internship 2025

Classes:
    - sertissage_conforme (correct crimp)
    - sertissage_non_conforme (defective crimp)
    - denudage_conforme (correct strip)
    - denudage_non_conforme (defective strip)

Usage:
    python src/train.py
"""

import os
import yaml
from pathlib import Path
from ultralytics import YOLO


def load_config(config_path: str = "config/training_config.yaml") -> dict:
    """Load training configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def download_dataset(api_key: str, workspace: str, project: str, version: int) -> str:
    """
    Download dataset from Roboflow.

    Args:
        api_key: Roboflow API key (set via ROBOFLOW_API_KEY env variable)
        workspace: Roboflow workspace name
        project: Roboflow project name
        version: Dataset version number

    Returns:
        Path to the downloaded dataset's data.yaml
    """
    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project)
    dataset = project.version(version).download("yolov8")
    return os.path.join(dataset.location, "data.yaml")


def train(config: dict, data_path: str) -> None:
    """Run YOLOv8 training with the specified configuration."""
    model = YOLO(config["model"])

    model.train(
        data=data_path,
        epochs=config["epochs"],
        imgsz=config["imgsz"],
        batch=config["batch"],
        lr0=config["lr0"],
        degrees=config["degrees"],
        mixup=config["mixup"],
        copy_paste=config["copy_paste"],
        patience=config.get("patience", 50),
    )

    print("\nTraining complete.")
    print(f"Best model saved to: runs/detect/train/weights/best.pt")


def export_ncnn(model_path: str = "runs/detect/train/weights/best.pt") -> None:
    """Export trained model to NCNN format for Raspberry Pi deployment."""
    model = YOLO(model_path)
    model.export(format="ncnn")
    print(f"NCNN model exported successfully.")


if __name__ == "__main__":
    config = load_config()

    # Download dataset from Roboflow (set your API key as env variable)
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if api_key:
        data_path = download_dataset(
            api_key=api_key,
            workspace="your-workspace",
            project="cable-defect-detection",
            version=7,
        )
    else:
        data_path = config["data"]
        print(f"No ROBOFLOW_API_KEY set. Using data path from config: {data_path}")

    train(config, data_path)

    # Uncomment to export after training:
    # export_ncnn()
