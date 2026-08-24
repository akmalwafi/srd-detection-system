import os

os.environ["KERAS_BACKEND"] = "tensorflow"

from pathlib import Path

import cv2
import keras
import keras_cv
import numpy as np
import yaml


# ============================================================
# SETTINGS
# ============================================================

IMAGE_PATH = Path(
    r"test_images\land2.jpg"
)

IMAGE_SIZE = 640

CONFIDENCE_THRESHOLD = 0.15


MODEL_CONFIGS = {
    "collar": {
        "model": Path(
            r"runs_tensorflow\detect\collar_full\weights\best.keras"
        ),
        "classes": Path(
            r"runs_tensorflow\detect\collar_full\classes.yaml"
        ),
    },

    "lanyard": {
        "model": Path(
            r"runs_tensorflow\detect\lanyard_clean_tf\weights\best.keras"
        ),
        "classes": Path(
            r"runs_tensorflow\detect\lanyard_clean_tf\classes.yaml"
        ),
    },

    "shoes": {
        "model": Path(
            r"runs_tensorflow\detect\shoes_full\weights\best.keras"
        ),
        "classes": Path(
            r"runs_tensorflow\detect\shoes_full\classes.yaml"
        ),
    },
}


OUTPUT_DIR = Path(
    "keras_test_results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# READ CLASS NAMES
# ============================================================

def load_class_names(class_file):

    if not class_file.exists():

        print(
            f"[WARNING] Class file not found:\n"
            f"{class_file}"
        )

        return []


    with open(
        class_file,
        "r",
        encoding="utf-8"
    ) as file:

        data = yaml.safe_load(
            file
        )


    names = data.get(
        "names",
        []
    )


    if isinstance(names, dict):

        names = [
            value
            for key, value
            in sorted(
                names.items(),
                key=lambda item: int(item[0])
            )
        ]


    return names


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image():

    image = cv2.imread(
        str(IMAGE_PATH)
    )


    if image is None:

        raise FileNotFoundError(
            f"Cannot read test image:\n"
            f"{IMAGE_PATH.resolve()}"
        )


    # Resize to same size used for training
    resized = cv2.resize(
        image,
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        )
    )


    # OpenCV = BGR
    # TensorFlow/Keras = RGB
    rgb = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2RGB
    )


    rgb = rgb.astype(
        np.float32
    )


    # Add batch dimension
    #
    # (640,640,3)
    #
    # becomes
    #
    # (1,640,640,3)

    batch = np.expand_dims(
        rgb,
        axis=0
    )


    return resized, batch


# ============================================================
# DRAW PREDICTIONS
# ============================================================

def draw_predictions(
    image,
    prediction,
    class_names,
    model_name
):

    output = image.copy()


    # KerasCV prediction dictionary
    boxes = prediction[
        "boxes"
    ][0]


    classes = prediction[
        "classes"
    ][0]


    confidences = prediction[
        "confidence"
    ][0]


    # Some KerasCV versions also return:
    #
    # num_detections

    num_detections = (
        prediction.get(
            "num_detections"
        )
    )


    if num_detections is not None:

        total = int(
            num_detections[0]
        )

        boxes = boxes[:total]
        classes = classes[:total]
        confidences = (
            confidences[:total]
        )


    print(
        f"\n{model_name.upper()} DETECTIONS"
    )

    print(
        "=" * 45
    )


    detected_count = 0


    for box, class_id, confidence in zip(
        boxes,
        classes,
        confidences
    ):

        confidence = float(
            confidence
        )


        if confidence < CONFIDENCE_THRESHOLD:

            continue


        class_id = int(
            class_id
        )


        # Ignore padding predictions
        if class_id < 0:
            continue


        if class_id < len(
            class_names
        ):

            class_name = (
                class_names[
                    class_id
                ]
            )

        else:

            class_name = (
                f"class_{class_id}"
            )


        x1, y1, x2, y2 = (
            box.astype(int)
        )


        # Keep boxes inside image
        x1 = max(
            0,
            min(
                x1,
                IMAGE_SIZE - 1
            )
        )

        y1 = max(
            0,
            min(
                y1,
                IMAGE_SIZE - 1
            )
        )

        x2 = max(
            0,
            min(
                x2,
                IMAGE_SIZE - 1
            )
        )

        y2 = max(
            0,
            min(
                y2,
                IMAGE_SIZE - 1
            )
        )


        detected_count += 1


        print(
            f"{detected_count}. "
            f"{class_name}"
            f" | confidence="
            f"{confidence:.2%}"
            f" | box="
            f"({x1},{y1},{x2},{y2})"
        )


        # Draw bounding box
        cv2.rectangle(
            output,
            (
                x1,
                y1
            ),
            (
                x2,
                y2
            ),
            (
                255,
                255,
                255
            ),
            2
        )


        label = (
            f"{class_name} "
            f"{confidence:.2f}"
        )


        cv2.putText(
            output,
            label,
            (
                x1,
                max(
                    y1 - 10,
                    20
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (
                255,
                255,
                255
            ),
            2
        )


    if detected_count == 0:

        print(
            "No detections above "
            f"{CONFIDENCE_THRESHOLD:.2f}"
        )


    return output


# ============================================================
# TEST ONE MODEL
# ============================================================

def test_model(
    model_name,
    model_config
):

    model_path = (
        model_config[
            "model"
        ]
    )


    classes_path = (
        model_config[
            "classes"
        ]
    )


    print(
        "\n\n=============================================="
    )

    print(
        f"TESTING {model_name.upper()}"
    )

    print(
        "=============================================="
    )


    if not model_path.exists():

        print(
            f"[ERROR] Model not found:\n"
            f"{model_path.resolve()}"
        )

        return


    print(
        f"Model:\n"
        f"{model_path.resolve()}"
    )


    # ========================================================
    # CLASS NAMES
    # ========================================================

    class_names = (
        load_class_names(
            classes_path
        )
    )


    print(
        f"Classes: "
        f"{class_names}"
    )


    # ========================================================
    # LOAD KERAS MODEL
    # ========================================================

    print(
        "\nLoading model..."
    )


    model = keras.models.load_model(
        model_path,
        compile=False
    )


    print(
        "Model loaded successfully."
    )


    # ========================================================
    # IMAGE
    # ========================================================

    image, batch = (
        load_image()
    )


    print(
        "\nRunning inference..."
    )


    # ========================================================
    # PREDICTION
    # ========================================================

    predictions = model.predict(
        batch,
        verbose=0
    )


    print(
        "\nPrediction keys:"
    )

    print(
        predictions.keys()
    )


    # ========================================================
    # DRAW RESULTS
    # ========================================================

    result_image = (
        draw_predictions(
            image,
            predictions,
            class_names,
            model_name
        )
    )


    # ========================================================
    # SAVE
    # ========================================================

    output_path = (
        OUTPUT_DIR /
        f"{model_name}_keras_result.jpg"
    )


    cv2.imwrite(
        str(output_path),
        result_image
    )


    print(
        f"\nSaved result:\n"
        f"{output_path.resolve()}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n=============================================="
    )

    print(
        "SRD TensorFlow / Keras Model Testing"
    )

    print(
        "=============================================="
    )


    print(
        f"\nTest image:\n"
        f"{IMAGE_PATH.resolve()}"
    )


    if not IMAGE_PATH.exists():

        raise FileNotFoundError(
            f"Test image does not exist:\n"
            f"{IMAGE_PATH.resolve()}"
        )


    for model_name, model_config in (
        MODEL_CONFIGS.items()
    ):

        try:

            test_model(
                model_name,
                model_config
            )

        except Exception as error:

            print(
                f"\n[ERROR] "
                f"{model_name} failed:"
            )

            print(
                error
            )


    print(
        "\n\n=============================================="
    )

    print(
        "ALL MODEL TESTS COMPLETE"
    )

    print(
        "=============================================="
    )


    print(
        f"\nResults folder:\n"
        f"{OUTPUT_DIR.resolve()}"
    )


if __name__ == "__main__":

    main()