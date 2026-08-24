import os

os.environ["KERAS_BACKEND"] = "tensorflow"

from pathlib import Path
import time
import csv

import cv2
import keras
import numpy as np
import yaml
import matplotlib.pyplot as plt
import keras_cv

# ============================================================
# SETTINGS
# ============================================================

MODELS = {

    "collar": {
        "model": Path(
            r"runs_tensorflow\detect\collar_full\weights\best.keras"
        ),
        "dataset": Path(
            r"collar.v1i.yolov8"
        ),
    },

    "lanyard": {
        "model": Path(
            r"runs_tensorflow\detect\lanyard_v2_tf\weights\best.keras"
        ),
        "dataset": Path(
            r"Lanyard_clean"
        ),
    },

    "shoes": {
        "model": Path(
            r"runs_tensorflow\detect\shoes_full\weights\best.keras"
        ),
        "dataset": Path(
            r"formal and non formal shoes.v1i.yolov8"
        ),
    },
}


IMAGE_SIZE = 640

# Used for Precision / Recall / F1 / confusion matrix
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.50

# For mAP50-95
MAP_IOU_THRESHOLDS = np.arange(
    0.50,
    0.96,
    0.05
)

OUTPUT_DIR = Path(
    "keras_evaluation_results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD CLASS NAMES
# ============================================================

def load_class_names(dataset_dir):

    yaml_path = (
        dataset_dir /
        "data.yaml"
    )

    with open(
        yaml_path,
        "r",
        encoding="utf-8"
    ) as file:

        data = yaml.safe_load(
            file
        )

    names = data["names"]

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
# IOU
# ============================================================

def calculate_iou(box1, box2):

    x1 = max(
        box1[0],
        box2[0]
    )

    y1 = max(
        box1[1],
        box2[1]
    )

    x2 = min(
        box1[2],
        box2[2]
    )

    y2 = min(
        box1[3],
        box2[3]
    )

    intersection_width = max(
        0,
        x2 - x1
    )

    intersection_height = max(
        0,
        y2 - y1
    )

    intersection = (
        intersection_width
        *
        intersection_height
    )

    area1 = max(
        0,
        box1[2] - box1[0]
    ) * max(
        0,
        box1[3] - box1[1]
    )

    area2 = max(
        0,
        box2[2] - box2[0]
    ) * max(
        0,
        box2[3] - box2[1]
    )

    union = (
        area1
        +
        area2
        -
        intersection
    )

    if union <= 0:
        return 0.0

    return (
        intersection /
        union
    )


# ============================================================
# READ YOLO GROUND TRUTH
# ============================================================

def read_ground_truth(label_path):

    objects = []

    if not label_path.exists():
        return objects

    with open(
        label_path,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            values = (
                line
                .strip()
                .split()
            )

            if len(values) < 5:
                continue

            class_id = int(
                float(
                    values[0]
                )
            )

            x_center = (
                float(values[1])
                * IMAGE_SIZE
            )

            y_center = (
                float(values[2])
                * IMAGE_SIZE
            )

            width = (
                float(values[3])
                * IMAGE_SIZE
            )

            height = (
                float(values[4])
                * IMAGE_SIZE
            )

            x1 = (
                x_center
                -
                width / 2
            )

            y1 = (
                y_center
                -
                height / 2
            )

            x2 = (
                x_center
                +
                width / 2
            )

            y2 = (
                y_center
                +
                height / 2
            )

            objects.append(
                {
                    "class_id": class_id,
                    "box": np.array(
                        [
                            x1,
                            y1,
                            x2,
                            y2
                        ],
                        dtype=np.float32
                    )
                }
            )

    return objects


# ============================================================
# PREPARE IMAGE
# ============================================================

def prepare_image(image_path):

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        raise ValueError(
            f"Cannot read image: "
            f"{image_path}"
        )

    image = cv2.resize(
        image,
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        )
    )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = image.astype(
        np.float32
    )

    return np.expand_dims(
        image,
        axis=0
    )


# ============================================================
# CONVERT KERAS PREDICTION
# ============================================================

def extract_predictions(
    prediction,
    minimum_confidence=0.0001
):

    boxes = (
        prediction["boxes"][0]
    )

    classes = (
        prediction["classes"][0]
    )

    confidences = (
        prediction["confidence"][0]
    )

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
        confidences = confidences[:total]

    results = []

    for (
        box,
        class_id,
        confidence
    ) in zip(
        boxes,
        classes,
        confidences
    ):

        class_id = int(
            class_id
        )

        confidence = float(
            confidence
        )

        if class_id < 0:
            continue

        if confidence < minimum_confidence:
            continue

        results.append(
            {
                "class_id": class_id,
                "confidence": confidence,
                "box": np.array(
                    box,
                    dtype=np.float32
                )
            }
        )

    return results


# ============================================================
# FIXED-THRESHOLD PRECISION / RECALL / F1
# ============================================================

def evaluate_fixed_threshold(
    predictions_by_image,
    ground_truth_by_image,
    number_of_classes
):

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for image_id in ground_truth_by_image.keys():

        predictions = [
            p
            for p in predictions_by_image[
                image_id
            ]
            if p["confidence"]
            >= CONF_THRESHOLD
        ]

        ground_truth = (
            ground_truth_by_image[
                image_id
            ]
        )

        matched_gt = set()

        predictions = sorted(
            predictions,
            key=lambda x:
            x["confidence"],
            reverse=True
        )

        for prediction in predictions:

            best_iou = 0
            best_gt_index = None

            for gt_index, gt in enumerate(
                ground_truth
            ):

                if gt_index in matched_gt:
                    continue

                if (
                    gt["class_id"]
                    !=
                    prediction["class_id"]
                ):
                    continue

                iou = calculate_iou(
                    prediction["box"],
                    gt["box"]
                )

                if iou > best_iou:

                    best_iou = iou
                    best_gt_index = (
                        gt_index
                    )

            if (
                best_gt_index is not None
                and
                best_iou >= IOU_THRESHOLD
            ):

                total_tp += 1

                matched_gt.add(
                    best_gt_index
                )

            else:

                total_fp += 1

        total_fn += (
            len(ground_truth)
            -
            len(matched_gt)
        )

    precision = (
        total_tp
        /
        (
            total_tp
            +
            total_fp
        )
        if (
            total_tp
            +
            total_fp
        ) > 0
        else 0
    )

    recall = (
        total_tp
        /
        (
            total_tp
            +
            total_fn
        )
        if (
            total_tp
            +
            total_fn
        ) > 0
        else 0
    )

    f1 = (
        2
        *
        precision
        *
        recall
        /
        (
            precision
            +
            recall
        )
        if (
            precision
            +
            recall
        ) > 0
        else 0
    )

    return (
        total_tp,
        total_fp,
        total_fn,
        precision,
        recall,
        f1
    )


# ============================================================
# AP CALCULATION
# ============================================================

def calculate_ap(
    recall,
    precision
):

    recall = np.concatenate(
        (
            [0.0],
            recall,
            [1.0]
        )
    )

    precision = np.concatenate(
        (
            [0.0],
            precision,
            [0.0]
        )
    )

    # Precision envelope
    for index in range(
        len(precision) - 2,
        -1,
        -1
    ):

        precision[index] = max(
            precision[index],
            precision[index + 1]
        )

    changing_points = np.where(
        recall[1:]
        !=
        recall[:-1]
    )[0]

    ap = np.sum(
        (
            recall[
                changing_points + 1
            ]
            -
            recall[
                changing_points
            ]
        )
        *
        precision[
            changing_points + 1
        ]
    )

    return float(ap)


# ============================================================
# AP FOR ONE CLASS AT ONE IOU
# ============================================================

def calculate_class_ap(
    class_id,
    iou_threshold,
    predictions_by_image,
    ground_truth_by_image
):

    total_ground_truth = 0

    gt_for_class = {}

    for image_id, objects in (
        ground_truth_by_image.items()
    ):

        boxes = [
            obj["box"]
            for obj in objects
            if obj["class_id"]
            == class_id
        ]

        gt_for_class[
            image_id
        ] = boxes

        total_ground_truth += (
            len(boxes)
        )

    if total_ground_truth == 0:
        return None

    predictions = []

    for image_id, objects in (
        predictions_by_image.items()
    ):

        for obj in objects:

            if (
                obj["class_id"]
                ==
                class_id
            ):

                predictions.append(
                    {
                        "image_id":
                            image_id,

                        "confidence":
                            obj[
                                "confidence"
                            ],

                        "box":
                            obj["box"]
                    }
                )

    predictions = sorted(
        predictions,
        key=lambda x:
        x["confidence"],
        reverse=True
    )

    matched = {
        image_id: set()
        for image_id
        in gt_for_class.keys()
    }

    tp = np.zeros(
        len(predictions)
    )

    fp = np.zeros(
        len(predictions)
    )

    for index, prediction in enumerate(
        predictions
    ):

        image_id = (
            prediction[
                "image_id"
            ]
        )

        gt_boxes = (
            gt_for_class.get(
                image_id,
                []
            )
        )

        best_iou = 0
        best_gt_index = None

        for gt_index, gt_box in enumerate(
            gt_boxes
        ):

            if (
                gt_index
                in matched[
                    image_id
                ]
            ):
                continue

            iou = calculate_iou(
                prediction["box"],
                gt_box
            )

            if iou > best_iou:

                best_iou = iou
                best_gt_index = (
                    gt_index
                )

        if (
            best_gt_index is not None
            and
            best_iou >= iou_threshold
        ):

            tp[index] = 1

            matched[
                image_id
            ].add(
                best_gt_index
            )

        else:

            fp[index] = 1

    tp_cumulative = (
        np.cumsum(tp)
    )

    fp_cumulative = (
        np.cumsum(fp)
    )

    recall = (
        tp_cumulative
        /
        total_ground_truth
    )

    precision = (
        tp_cumulative
        /
        np.maximum(
            tp_cumulative
            +
            fp_cumulative,
            1e-10
        )
    )

    return calculate_ap(
        recall,
        precision
    )


# ============================================================
# CALCULATE mAP
# ============================================================

def calculate_map(
    predictions_by_image,
    ground_truth_by_image,
    number_of_classes
):

    class_ap50 = {}

    class_ap50_95 = {}

    all_ap50 = []
    all_ap50_95 = []

    for class_id in range(
        number_of_classes
    ):

        ap_values = []

        for iou_threshold in (
            MAP_IOU_THRESHOLDS
        ):

            ap = calculate_class_ap(
                class_id,
                iou_threshold,
                predictions_by_image,
                ground_truth_by_image
            )

            if ap is not None:
                ap_values.append(
                    ap
                )

        ap50 = calculate_class_ap(
            class_id,
            0.50,
            predictions_by_image,
            ground_truth_by_image
        )

        if ap50 is not None:

            class_ap50[
                class_id
            ] = ap50

            all_ap50.append(
                ap50
            )

        if len(ap_values) > 0:

            mean_ap = float(
                np.mean(
                    ap_values
                )
            )

            class_ap50_95[
                class_id
            ] = mean_ap

            all_ap50_95.append(
                mean_ap
            )

    map50 = (
        float(
            np.mean(
                all_ap50
            )
        )
        if all_ap50
        else 0.0
    )

    map50_95 = (
        float(
            np.mean(
                all_ap50_95
            )
        )
        if all_ap50_95
        else 0.0
    )

    return (
        map50,
        map50_95,
        class_ap50,
        class_ap50_95
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

def build_confusion_matrix(
    predictions_by_image,
    ground_truth_by_image,
    number_of_classes
):

    # Extra class = background
    background_id = (
        number_of_classes
    )

    matrix = np.zeros(
        (
            number_of_classes + 1,
            number_of_classes + 1
        ),
        dtype=np.int32
    )

    for image_id in ground_truth_by_image:

        predictions = [
            prediction
            for prediction
            in predictions_by_image[
                image_id
            ]
            if prediction[
                "confidence"
            ] >= CONF_THRESHOLD
        ]

        ground_truth = (
            ground_truth_by_image[
                image_id
            ]
        )

        candidates = []

        for pred_index, pred in enumerate(
            predictions
        ):

            for gt_index, gt in enumerate(
                ground_truth
            ):

                iou = calculate_iou(
                    pred["box"],
                    gt["box"]
                )

                if iou >= IOU_THRESHOLD:

                    candidates.append(
                        (
                            iou,
                            pred_index,
                            gt_index
                        )
                    )

        candidates.sort(
            reverse=True,
            key=lambda x:
            x[0]
        )

        matched_predictions = set()
        matched_ground_truth = set()

        for (
            iou,
            pred_index,
            gt_index
        ) in candidates:

            if (
                pred_index
                in matched_predictions
            ):
                continue

            if (
                gt_index
                in matched_ground_truth
            ):
                continue

            pred_class = (
                predictions[
                    pred_index
                ][
                    "class_id"
                ]
            )

            gt_class = (
                ground_truth[
                    gt_index
                ][
                    "class_id"
                ]
            )

            if (
                pred_class
                < number_of_classes
                and
                gt_class
                < number_of_classes
            ):

                # Rows = predicted
                # Columns = actual
                matrix[
                    pred_class,
                    gt_class
                ] += 1

            matched_predictions.add(
                pred_index
            )

            matched_ground_truth.add(
                gt_index
            )

        # False positives
        for pred_index, pred in enumerate(
            predictions
        ):

            if (
                pred_index
                not in matched_predictions
            ):

                pred_class = (
                    pred[
                        "class_id"
                    ]
                )

                if (
                    pred_class
                    < number_of_classes
                ):

                    matrix[
                        pred_class,
                        background_id
                    ] += 1

        # False negatives
        for gt_index, gt in enumerate(
            ground_truth
        ):

            if (
                gt_index
                not in matched_ground_truth
            ):

                gt_class = (
                    gt[
                        "class_id"
                    ]
                )

                if (
                    gt_class
                    < number_of_classes
                ):

                    matrix[
                        background_id,
                        gt_class
                    ] += 1

    return matrix


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

def save_confusion_matrix(
    matrix,
    class_names,
    output_path,
    title
):

    labels = (
        class_names
        +
        ["background"]
    )

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    image = ax.imshow(
        matrix
    )

    fig.colorbar(
        image,
        ax=ax
    )

    ax.set_xticks(
        range(
            len(labels)
        )
    )

    ax.set_yticks(
        range(
            len(labels)
        )
    )

    ax.set_xticklabels(
        labels,
        rotation=45,
        ha="right"
    )

    ax.set_yticklabels(
        labels
    )

    ax.set_xlabel(
        "True Class"
    )

    ax.set_ylabel(
        "Predicted Class"
    )

    ax.set_title(
        title
    )

    for row in range(
        matrix.shape[0]
    ):

        for column in range(
            matrix.shape[1]
        ):

            ax.text(
                column,
                row,
                str(
                    matrix[
                        row,
                        column
                    ]
                ),
                ha="center",
                va="center"
            )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200
    )

    plt.close()


# ============================================================
# EVALUATE ONE MODEL
# ============================================================

def evaluate_model(
    model_name,
    config
):

    print(
        "\n\n=============================================="
    )

    print(
        f"EVALUATING: "
        f"{model_name.upper()}"
    )

    print(
        "=============================================="
    )

    model_path = (
        config[
            "model"
        ]
    )

    dataset_dir = (
        config[
            "dataset"
        ]
    )

    test_images = (
        dataset_dir
        /
        "test"
        /
        "images"
    )

    test_labels = (
        dataset_dir
        /
        "test"
        /
        "labels"
    )

    if not model_path.exists():

        raise FileNotFoundError(
            f"Model not found:\n"
            f"{model_path.resolve()}"
        )

    if not test_images.exists():

        raise FileNotFoundError(
            f"Test images not found:\n"
            f"{test_images.resolve()}"
        )

    class_names = (
        load_class_names(
            dataset_dir
        )
    )

    number_of_classes = (
        len(
            class_names
        )
    )

    print(
        f"Model:   "
        f"{model_path}"
    )

    print(
        f"Dataset: "
        f"{dataset_dir}"
    )

    print(
        f"Classes: "
        f"{class_names}"
    )

    print(
        "\nLoading model..."
    )

    model = (
        keras.models.load_model(
            model_path,
            compile=False
        )
    )

    # Use a very low decoder threshold so AP/mAP
    # evaluation can inspect low-confidence predictions.
    model.prediction_decoder = (
        keras_cv.layers.NonMaxSuppression(
            bounding_box_format=model.bounding_box_format,
            from_logits=False,
            confidence_threshold=0.001,
            iou_threshold=0.7
        )
    )

    print(
        "Model loaded successfully."
    )

    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp"
    }

    image_paths = sorted(
        [
            path
            for path
            in test_images.iterdir()
            if path.suffix.lower()
            in extensions
        ]
    )

    print(
        f"\nTest images: "
        f"{len(image_paths)}"
    )

    predictions_by_image = {}

    ground_truth_by_image = {}

    inference_times = []

    for index, image_path in enumerate(
        image_paths,
        start=1
    ):

        batch = (
            prepare_image(
                image_path
            )
        )

        label_path = (
            test_labels
            /
            f"{image_path.stem}.txt"
        )

        ground_truth = (
            read_ground_truth(
                label_path
            )
        )

        # Warm-up is naturally handled after
        # first prediction, but every image
        # timing is recorded here.

        start_time = (
            time.perf_counter()
        )

        prediction = (
            model.predict(
                batch,
                verbose=0
            )
        )

        if model_name == "lanyard":

            boxes = prediction["boxes"][0]
            classes = prediction["classes"][0]
            confidences = prediction["confidence"][0]

            order = np.argsort(confidences)[::-1]

            print("\nTOP LANYARD PREDICTIONS:")

            for prediction_index in order[:5]:
                confidence = float(
                    confidences[prediction_index]
                )

                class_id = int(
                    classes[prediction_index]
                )

                box = boxes[prediction_index]

                print(
                    f"class={class_id} "
                    f"confidence={confidence:.4f} "
                    f"box={box}"
                )

            print(
                "Ground truth classes:",
                [
                    gt["class_id"]
                    for gt in ground_truth
                ]
            )

        end_time = (
            time.perf_counter()
        )

        inference_ms = (
            (
                end_time
                -
                start_time
            )
            *
            1000
        )

        inference_times.append(
            inference_ms
        )

        detections = (
            extract_predictions(
                prediction
            )
        )

        image_id = (
            image_path.name
        )

        predictions_by_image[
            image_id
        ] = detections

        ground_truth_by_image[
            image_id
        ] = ground_truth

        print(
            f"[{index}/{len(image_paths)}] "
            f"{image_path.name} "
            f"| GT={len(ground_truth)} "
            f"| Predictions="
            f"{len(detections)}"
        )

    # ========================================================
    # PRECISION / RECALL / F1
    # ========================================================

    (
        tp,
        fp,
        fn,
        precision,
        recall,
        f1
    ) = evaluate_fixed_threshold(
        predictions_by_image,
        ground_truth_by_image,
        number_of_classes
    )

    # ========================================================
    # mAP
    # ========================================================

    (
        map50,
        map50_95,
        class_ap50,
        class_ap50_95
    ) = calculate_map(
        predictions_by_image,
        ground_truth_by_image,
        number_of_classes
    )

    # ========================================================
    # SPEED
    # ========================================================

    average_ms = float(
        np.mean(
            inference_times
        )
    )

    fps = (
        1000
        /
        average_ms
        if average_ms > 0
        else 0
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    matrix = (
        build_confusion_matrix(
            predictions_by_image,
            ground_truth_by_image,
            number_of_classes
        )
    )

    model_output_dir = (
        OUTPUT_DIR
        /
        model_name
    )

    model_output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    save_confusion_matrix(
        matrix,
        class_names,
        model_output_dir
        /
        "confusion_matrix.png",
        f"{model_name.upper()} "
        f"Confusion Matrix"
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print(
        "\n----------------------------------------------"
    )

    print(
        f"{model_name.upper()} RESULTS"
    )

    print(
        "----------------------------------------------"
    )

    print(
        f"True Positives:  "
        f"{tp}"
    )

    print(
        f"False Positives: "
        f"{fp}"
    )

    print(
        f"False Negatives: "
        f"{fn}"
    )

    print()

    print(
        f"Precision:   "
        f"{precision:.4f} "
        f"({precision * 100:.2f}%)"
    )

    print(
        f"Recall:      "
        f"{recall:.4f} "
        f"({recall * 100:.2f}%)"
    )

    print(
        f"F1 Score:    "
        f"{f1:.4f} "
        f"({f1 * 100:.2f}%)"
    )

    print(
        f"mAP50:       "
        f"{map50:.4f} "
        f"({map50 * 100:.2f}%)"
    )

    print(
        f"mAP50-95:    "
        f"{map50_95:.4f} "
        f"({map50_95 * 100:.2f}%)"
    )

    print(
        f"Average inference: "
        f"{average_ms:.2f} ms/image"
    )

    print(
        f"FPS: "
        f"{fps:.2f}"
    )

    print(
        "\nPer-class AP:"
    )

    for class_id, class_name in enumerate(
        class_names
    ):

        ap50 = (
            class_ap50.get(
                class_id,
                0
            )
        )

        ap5095 = (
            class_ap50_95.get(
                class_id,
                0
            )
        )

        print(
            f"{class_name}: "
            f"mAP50={ap50:.4f} | "
            f"mAP50-95={ap5095:.4f}"
        )

    print(
        "\nConfusion Matrix:"
    )

    print(
        matrix
    )

    print(
        "\nSaved:"
    )

    print(
        model_output_dir.resolve()
    )

    return {
        "Model": model_name,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "mAP50": map50,
        "mAP50-95": map50_95,
        "Inference_ms": average_ms,
        "FPS": fps
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n=============================================="
    )

    print(
        "SRD TENSORFLOW / KERAS EVALUATION"
    )

    print(
        "=============================================="
    )

    summary = []

    for model_name, config in (
        MODELS.items()
    ):

        try:

            result = (
                evaluate_model(
                    model_name,
                    config
                )
            )

            summary.append(
                result
            )

        except Exception as error:

            print(
                f"\n[ERROR] "
                f"{model_name} failed:"
            )

            print(
                error
            )

    # ========================================================
    # SAVE SUMMARY CSV
    # ========================================================

    if not summary:
        return

    summary_path = (
        OUTPUT_DIR
        /
        "keras_metrics_summary.csv"
    )

    with open(
        summary_path,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = (
            csv.DictWriter(
                file,
                fieldnames=
                summary[0].keys()
            )
        )

        writer.writeheader()

        writer.writerows(
            summary
        )

    # ========================================================
    # FINAL TABLE
    # ========================================================

    print(
        "\n\n=============================================="
    )

    print(
        "FINAL KERAS COMPARISON"
    )

    print(
        "=============================================="
    )

    print(
        f"\n"
        f"{'Model':<12}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1':>12}"
        f"{'mAP50':>12}"
        f"{'mAP50-95':>14}"
        f"{'FPS':>10}"
    )

    for row in summary:

        print(
            f"{row['Model']:<12}"
            f"{row['Precision']:>12.4f}"
            f"{row['Recall']:>12.4f}"
            f"{row['F1']:>12.4f}"
            f"{row['mAP50']:>12.4f}"
            f"{row['mAP50-95']:>14.4f}"
            f"{row['FPS']:>10.2f}"
        )

    print(
        f"\nSummary saved to:\n"
        f"{summary_path.resolve()}"
    )


if __name__ == "__main__":
    main()