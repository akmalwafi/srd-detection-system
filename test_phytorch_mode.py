from pathlib import Path

import torch
from ultralytics import YOLO


# Every model uses the ORIGINAL PyTorch .pt weights
MODEL_PATHS = {
    "collar": Path(
        r"runs\detect\train4\weights\best.pt"
    ),
    "lanyard": Path(
        r"runs\detect\train7\weights\best.pt"
    ),
    "shoes": Path(
        r"runs\detect\train8\weights\best.pt"
    )
}

# SAME image used for TensorFlow testing
IMAGE_PATH = Path(
    r"test_images\shocollar.jpg"
)

OUTPUT_DIR = Path("pytorch_results")


def main() -> None:
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Test image not found:\n{IMAGE_PATH.resolve()}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for model_name, model_path in MODEL_PATHS.items():

        print(f"\n{'=' * 50}")
        print(f"Testing {model_name.upper()} model")
        print(f"Model path: {model_path.resolve()}")

        if not model_path.exists():
            print(
                f"[SKIPPED] PyTorch model not found:\n"
                f"{model_path.resolve()}"
            )
            continue

        try:

            # Load original PyTorch YOLO model
            model = YOLO(
                str(model_path)
            )

            print("Classes:", model.names)

            # IMPORTANT:
            # Keep SAME parameters as TensorFlow test
            results = model.predict(
                source=str(IMAGE_PATH),
                conf=0.15,
                imgsz=640,
                verbose=True
            )

            if not results:
                print(f"No result returned by {model_name}.")
                continue

            result = results[0]

            if result.boxes is not None and len(result.boxes) > 0:

                print(f"\n{model_name.capitalize()} detections:")

                for box in result.boxes:

                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    class_name = result.names[class_id]

                    x1, y1, x2, y2 = map(
                        int,
                        box.xyxy[0].tolist()
                    )

                    print(
                        f"- {class_name}: "
                        f"{confidence:.2%} "
                        f"| box=({x1}, {y1}, {x2}, {y2})"
                    )

            else:
                print(f"No {model_name} detected.")

            output_path = (
                OUTPUT_DIR /
                f"{model_name}_pytorch_result.jpg"
            )

            result.save(
                filename=str(output_path)
            )

            print(
                f"Saved result: {output_path.resolve()}"
            )

        except Exception as error:

            print(
                f"[ERROR] Failed to run {model_name}: "
                f"{error}"
            )

    print(
        "\nAll available PyTorch models have been tested."
    )


if __name__ == "__main__":
    main()