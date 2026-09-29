"""Colab / Remote GPU Training Script for YOLOv11n on MapleStory Lotus Phase 1 Synthetic Vision Dataset.

Trains an Ultralytics YOLOv11n lightweight object detector on the auto-generated
synthetic dataset, validates mAP, and exports to PyTorch (.pt) and ONNX formats.
"""

import argparse
import os
import sys

def train_yolo(
    data_yaml: str = "dataset_yolo/data.yaml",
    model_name: str = "yolo11n.pt",
    epochs: int = 50,
    imgsz: int = 640,
    batch_size: int = 32,
    project: str = "runs/lotus_vision",
    name: str = "yolo11n_phase1",
):
    print(f"[*] Starting YOLO Training Pipeline...")
    print(f"    Model      : {model_name}")
    print(f"    Data Config: {data_yaml}")
    print(f"    Epochs     : {epochs}")
    print(f"    Image Size : {imgsz}")
    print(f"    Batch Size : {batch_size}")

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[!] Ultralytics not found. Installing 'ultralytics'...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics"])
        from ultralytics import YOLO

    import torch
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"    Device     : {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"Data config not found: {data_yaml}. Run export_yolo_dataset.py first!")

    # Load pretrained YOLO model
    print(f"[*] Loading base model '{model_name}'...")
    model = YOLO(model_name)

    # Train
    print(f"[*] Beginning training for {epochs} epochs...")
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch_size,
        device=device,
        project=project,
        name=name,
        workers=4,
        save=True,
        verbose=True,
    )

    # Validation
    print(f"\n[*] Evaluating model on validation set...")
    metrics = model.val()
    print(f"    mAP@50     : {metrics.box.map50:.4f}")
    print(f"    mAP@50-95  : {metrics.box.map:.4f}")

    # Export to ONNX
    print(f"\n[*] Exporting best model to ONNX for high-speed inference...")
    onnx_path = model.export(format="onnx", half=(device != "cpu"))
    print(f"[SUCCESS] YOLO model trained and exported successfully!")
    print(f"          Weights: {project}/{name}/weights/best.pt")
    print(f"          ONNX   : {onnx_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLOv11 on Lotus Phase 1 Synthetic Vision Dataset")
    parser.add_argument("--data_yaml", type=str, default="dataset_yolo/data.yaml", help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Base model (yolo11n.pt or yolov8n.pt)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution for training")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size")
    parser.add_argument("--project", type=str, default="runs/lotus_vision", help="Project output folder")
    parser.add_argument("--name", type=str, default="yolo11n_phase1", help="Run experiment name")

    args = parser.parse_args()
    train_yolo(
        data_yaml=args.data_yaml,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch_size=args.batch_size,
        project=args.project,
        name=args.name,
    )
