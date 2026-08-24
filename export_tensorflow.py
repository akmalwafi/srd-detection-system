import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path

import tensorflow as tf
from ultralytics import YOLO


MODEL_PATHS = {
    "collar": Path("runs/detect/train4/weights/best.pt"),
    "lanyard": Path("runs/detect/train7/weights/best.pt"),
    "shoes": Path("runs/detect/train8/weights/best.pt"),
}


def export_model(model_name: str, model_path: Path) -> None:
    if not model_path.exists():
        print(f"[ERROR] {model_name} model not found:")
        print(model_path)
        return

    print(f"\nExporting {model_name} model...")

    # Load the existing YOLO PyTorch model
    model = YOLO(str(model_path))

    # Convert it to TensorFlow SavedModel
    exported_path = model.export(
        format="saved_model",
        imgsz=640,
        device="cpu"
    )

    print(f"{model_name} exported to:")
    print(exported_path)

    # Verify that TensorFlow can load the model
    tensorflow_model = tf.saved_model.load(str(exported_path))

    print(
        "TensorFlow signatures:",
        list(tensorflow_model.signatures.keys())
    )


def main() -> None:
    print("TensorFlow version:", tf.__version__)

    for model_name, model_path in MODEL_PATHS.items():
        export_model(model_name, model_path)


if __name__ == "__main__":
    main()