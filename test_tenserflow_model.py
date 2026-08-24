from pathlib import Path

import tensorflow as tf
from ultralytics import YOLO


# Every model has its own TensorFlow SavedModel folder
MODEL_PATHS = {
    "collar": Path(
        r"runs\detect\train4\weights\best_saved_model"
    ),
    "lanyard": Path(
        r"runs\detect\train7\weights\best_saved_model"
    ),
    "shoes": Path(
        r"runs\detect\train8\weights\best_saved_model"
    )
}

IMAGE_PATH = Path(
    r"test_images\land2.jpg"
)

OUTPUT_DIR = Path("tensorflow_results")


def main() -> None:
    print("TensorFlow version:", tf.__version__)

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
                f"[SKIPPED] TensorFlow model not found:\n"
                f"{model_path.resolve()}"
            )
            continue

        try:
            # Loads the TensorFlow SavedModel through Ultralytics
            model = YOLO(
                str(model_path),
                task="detect"
            )

            print("Classes:", model.names)

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

                    print(
                        f"- {class_name}: "
                        f"{confidence:.2%}"
                    )
            else:
                print(f"No {model_name} detected.")

            output_path = (
                OUTPUT_DIR /
                f"{model_name}_tensorflow_result.jpg"
            )

            result.save(filename=str(output_path))

            print(f"Saved result: {output_path.resolve()}")

        except Exception as error:
            print(
                f"[ERROR] Failed to run {model_name}: "
                f"{error}"
            )

    print("\nAll available TensorFlow models have been tested.")


if __name__ == "__main__":
    main()