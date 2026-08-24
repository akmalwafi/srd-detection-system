import os

# IMPORTANT: must be set before importing Keras
os.environ["KERAS_BACKEND"] = "tensorflow"

import argparse
from pathlib import Path

import keras
import keras_cv
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import yaml
from PIL import Image


# ============================================================
# SETTINGS
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}


# ============================================================
# READ CLASS NAMES FROM data.yaml
# ============================================================

def read_class_names(dataset_dir: Path):

    yaml_path = dataset_dir / "data.yaml"

    if not yaml_path.exists():
        raise FileNotFoundError(
            f"data.yaml not found:\n{yaml_path.resolve()}"
        )

    with open(
        yaml_path,
        "r",
        encoding="utf-8"
    ) as file:
        data = yaml.safe_load(file)

    names = data.get("names")

    if names is None:
        raise ValueError(
            "No 'names' entry found in data.yaml"
        )

    # Roboflow sometimes stores names as a list
    if isinstance(names, list):
        return names

    # Or as:
    # names:
    #   0: collar
    #   1: nocollar
    if isinstance(names, dict):

        sorted_items = sorted(
            names.items(),
            key=lambda x: int(x[0])
        )

        return [
            str(name)
            for _, name in sorted_items
        ]

    raise ValueError(
        "Unsupported class format in data.yaml"
    )


# ============================================================
# FIND IMAGES
# ============================================================

def get_image_paths(images_dir: Path):

    if not images_dir.exists():
        raise FileNotFoundError(
            f"Images folder not found:\n"
            f"{images_dir.resolve()}"
        )

    paths = [
        path
        for path in images_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    return sorted(paths)


# ============================================================
# READ YOLO BOUNDING BOX LABELS
# ============================================================

def read_yolo_label(
    label_path: Path,
    image_width: int,
    image_height: int
):

    boxes = []
    classes = []

    if not label_path.exists():
        return boxes, classes

    with open(
        label_path,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()

            if not line:
                continue

            values = line.split()

            if len(values) < 5:
                continue

            class_id = int(float(values[0]))

            x_center = float(values[1])
            y_center = float(values[2])
            box_width = float(values[3])
            box_height = float(values[4])

            # YOLO values are normalized
            x_center *= image_width
            y_center *= image_height
            box_width *= image_width
            box_height *= image_height

            # Convert:
            #
            # YOLO center_xywh
            #
            # to:
            #
            # XYXY
            #

            x1 = x_center - box_width / 2
            y1 = y_center - box_height / 2

            x2 = x_center + box_width / 2
            y2 = y_center + box_height / 2

            boxes.append(
                [
                    x1,
                    y1,
                    x2,
                    y2
                ]
            )

            classes.append(
                class_id
            )

    return boxes, classes


# ============================================================
# LOAD A DATASET SPLIT
# ============================================================

def scan_dataset_split(
    dataset_dir: Path,
    split_name: str
):

    images_dir = (
        dataset_dir /
        split_name /
        "images"
    )

    labels_dir = (
        dataset_dir /
        split_name /
        "labels"
    )

    image_paths = get_image_paths(
        images_dir
    )

    all_boxes = []
    all_classes = []
    valid_image_paths = []

    print(
        f"\nScanning {split_name}: "
        f"{len(image_paths)} images"
    )

    for image_path in image_paths:

        with Image.open(image_path) as image:

            width, height = (
                image.size
            )

        label_path = (
            labels_dir /
            f"{image_path.stem}.txt"
        )

        boxes, classes = (
            read_yolo_label(
                label_path,
                width,
                height
            )
        )

        valid_image_paths.append(
            str(image_path)
        )

        all_boxes.append(
            boxes
        )

        all_classes.append(
            classes
        )

    return (
        valid_image_paths,
        all_boxes,
        all_classes
    )


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image(image_path):

    image_bytes = tf.io.read_file(
        image_path
    )

    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False
    )

    image.set_shape(
        [
            None,
            None,
            3
        ]
    )

    return tf.cast(
        image,
        tf.float32
    )


# ============================================================
# CREATE TF.DATA DATASET
# ============================================================

def create_tf_dataset(
    image_paths,
    boxes,
    classes,
    batch_size,
    image_size,
    training
):

    image_paths = tf.constant(
        image_paths
    )

    boxes = tf.ragged.constant(
        boxes,
        dtype=tf.float32
    )

    classes = tf.ragged.constant(
        classes,
        dtype=tf.float32
    )

    dataset = (
        tf.data.Dataset
        .from_tensor_slices(
            (
                image_paths,
                classes,
                boxes
            )
        )
    )


    def load_sample(
        image_path,
        sample_classes,
        sample_boxes
    ):

        image = load_image(
            image_path
        )

        bounding_boxes = {
            "classes": sample_classes,
            "boxes": sample_boxes
        }

        return {
            "images": image,
            "bounding_boxes": bounding_boxes
        }


    dataset = dataset.map(
        load_sample,
        num_parallel_calls=tf.data.AUTOTUNE
    )


    if training:

        dataset = dataset.shuffle(
            max(
                len(image_paths),
                100
            )
        )


    dataset = dataset.ragged_batch(
        batch_size,
        drop_remainder=False
    )


    # ========================================================
    # AUGMENTATION
    # ========================================================

    if training:

        transform = keras.Sequential(
            [
                keras_cv.layers.RandomFlip(
                    mode="horizontal",
                    bounding_box_format="xyxy"
                ),

                keras_cv.layers.JitteredResize(
                    target_size=(
                        image_size,
                        image_size
                    ),

                    scale_factor=(
                        0.8,
                        1.2
                    ),

                    bounding_box_format="xyxy"
                )
            ]
        )

    else:

        transform = (
            keras_cv.layers.JitteredResize(
                target_size=(
                    image_size,
                    image_size
                ),

                scale_factor=(
                    1.0,
                    1.0
                ),

                bounding_box_format="xyxy"
            )
        )


    dataset = dataset.map(
        transform,
        num_parallel_calls=tf.data.AUTOTUNE
    )


    def dict_to_tuple(data):

        return (
            data["images"],
            data["bounding_boxes"]
        )


    dataset = dataset.map(
        dict_to_tuple,
        num_parallel_calls=tf.data.AUTOTUNE
    )


    return dataset.prefetch(
        tf.data.AUTOTUNE
    )


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

def create_run_directory(
    experiment_name
):

    root = (
        Path("runs_tensorflow")
        / "detect"
    )

    run_dir = (
        root /
        experiment_name
    )

    original = run_dir

    number = 2

    # Do not overwrite an older experiment
    while run_dir.exists():

        run_dir = Path(
            f"{original}{number}"
        )

        number += 1

    weights_dir = (
        run_dir /
        "weights"
    )

    weights_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    return (
        run_dir,
        weights_dir
    )


# ============================================================
# MAIN TRAINING
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
        type=str
    )

    parser.add_argument(
        "--name",
        required=True,
        type=str
    )

    parser.add_argument(
        "--epochs",
        default=50,
        type=int
    )

    parser.add_argument(
        "--batch",
        default=4,
        type=int
    )

    parser.add_argument(
        "--imgsz",
        default=640,
        type=int
    )

    args = parser.parse_args()


    dataset_dir = Path(
        args.dataset
    )


    # ========================================================
    # ENVIRONMENT INFO
    # ========================================================

    print(
        "\n====================================="
    )

    print(
        "TensorFlow / Keras Object Detection"
    )

    print(
        "====================================="
    )

    print(
        "TensorFlow:",
        tf.__version__
    )

    print(
        "Keras:",
        keras.__version__
    )

    print(
        "GPU devices:",
        tf.config.list_physical_devices(
            "GPU"
        )
    )


    # ========================================================
    # CLASSES
    # ========================================================

    class_names = (
        read_class_names(
            dataset_dir
        )
    )

    print(
        "\nClasses:"
    )

    for class_id, class_name in enumerate(
        class_names
    ):

        print(
            f"{class_id}: "
            f"{class_name}"
        )


    # ========================================================
    # DATA
    # ========================================================

    (
        train_images,
        train_boxes,
        train_classes
    ) = scan_dataset_split(
        dataset_dir,
        "train"
    )


    (
        val_images,
        val_boxes,
        val_classes
    ) = scan_dataset_split(
        dataset_dir,
        "valid"
    )


    train_dataset = (
        create_tf_dataset(
            train_images,
            train_boxes,
            train_classes,
            args.batch,
            args.imgsz,
            training=True
        )
    )


    val_dataset = (
        create_tf_dataset(
            val_images,
            val_boxes,
            val_classes,
            args.batch,
            args.imgsz,
            training=False
        )
    )


    # ========================================================
    # MODEL
    # ========================================================

    print(
        "\nCreating KerasCV YOLOv8 model..."
    )


    backbone = (
        keras_cv.models
        .YOLOV8Backbone
        .from_preset(
            "yolo_v8_xs_backbone_coco"
        )
    )


    model = (
        keras_cv.models
        .YOLOV8Detector(
            num_classes=len(
                class_names
            ),

            bounding_box_format="xyxy",

            backbone=backbone,

            fpn_depth=1
        )
    )


    optimizer = (
        keras.optimizers.Adam(
            learning_rate=0.001,
            global_clipnorm=10.0
        )
    )


    model.compile(
        optimizer=optimizer,

        classification_loss=(
            "binary_crossentropy"
        ),

        box_loss="ciou"
    )


    # ========================================================
    # OUTPUT
    # ========================================================

    run_dir, weights_dir = (
        create_run_directory(
            args.name
        )
    )


    print(
        f"\nSaving run to:\n"
        f"{run_dir.resolve()}"
    )


    # Save class list
    with open(
        run_dir / "classes.yaml",
        "w",
        encoding="utf-8"
    ) as file:

        yaml.safe_dump(
            {
                "names": class_names
            },
            file
        )


    # ========================================================
    # CALLBACKS
    # ========================================================

    checkpoint = (
        keras.callbacks.ModelCheckpoint(
            filepath=str(
                weights_dir /
                "best.weights.h5"
            ),

            monitor="val_loss",

            save_best_only=True,

            save_weights_only=True,

            verbose=1
        )
    )


    early_stopping = (
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
            verbose=1
        )
    )


    csv_logger = (
        keras.callbacks.CSVLogger(
            str(
                run_dir /
                "history.csv"
            )
        )
    )


    # ========================================================
    # TRAIN
    # ========================================================

    history = model.fit(
        train_dataset,

        validation_data=(
            val_dataset
        ),

        epochs=args.epochs,

        callbacks=[
            checkpoint,
            early_stopping,
            csv_logger
        ]
    )


    # ========================================================
    # LOAD BEST WEIGHTS
    # ========================================================

    best_weights = (
        weights_dir /
        "best.weights.h5"
    )

    if best_weights.exists():

        model.load_weights(
            best_weights
        )


    # ========================================================
    # SAVE COMPLETE MODEL
    # ========================================================

    model_path = (
        weights_dir /
        "best.keras"
    )

    model.save(
        model_path
    )


    # ========================================================
    # LOSS GRAPH
    # ========================================================

    history_data = (
        history.history
    )


    plt.figure(
        figsize=(10, 6)
    )


    if "loss" in history_data:

        plt.plot(
            history_data["loss"],
            label="Training loss"
        )


    if "val_loss" in history_data:

        plt.plot(
            history_data["val_loss"],
            label="Validation loss"
        )


    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        f"{args.name} - TensorFlow/Keras"
    )

    plt.legend()

    plt.grid()


    plt.savefig(
        run_dir /
        "results.png",
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()


    print(
        "\n====================================="
    )

    print(
        "TRAINING COMPLETE"
    )

    print(
        "====================================="
    )

    print(
        f"Best model:\n"
        f"{model_path.resolve()}"
    )


if __name__ == "__main__":
    main()