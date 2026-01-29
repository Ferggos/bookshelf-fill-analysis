import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Train YOLO11 segmentation on local dataset")
    parser.add_argument("--data", default="dataset/data.yaml", help="Path to data.yaml")
    parser.add_argument("--model", default="yolo11n-seg.pt", help="Base model weights")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=150, help="Number of epochs")
    parser.add_argument("--workers", type=int, default=2, help="Dataloader workers")
    parser.add_argument("--device", default="0", help="Device id or 'cpu'")
    parser.add_argument("--close-mosaic", type=int, default=10, help="Close mosaic N epochs before end")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_path}")

    device = args.device
    if device != "cpu" and not torch.cuda.is_available():
        device = "cpu"

    model = YOLO(args.model)

    model.train(
        data=str(data_path),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=device,
        workers=args.workers,
        amp=True,
        cache=False,
        close_mosaic=args.close_mosaic,
        optimizer="auto",
        patience=20,
    )


if __name__ == "__main__":
    main()
