from pathlib import Path
import shutil
import yaml


# ============================================================
# CONFIGURATION
# ============================================================

SOURCE_DATASET = Path(
    "Lanyard3.v3i.yolov8"
)

OUTPUT_DATASET = Path(
    "Lanyard_clean"
)


# Original class IDs that we want to KEEP.
#
# According to your current data.yaml:
# class 5 = ID-CARD
KEEP_CLASS_IDS = {
    5
}


# All kept classes will become:
#
# 0 = lanyard
NEW_CLASS_NAME = "lanyard"


SPLITS = [
    "train",
    "valid",
    "test"
]


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}


# ============================================================
# CREATE FOLDERS
# ============================================================

def create_folders():

    for split in SPLITS:

        (
            OUTPUT_DATASET /
            split /
            "images"
        ).mkdir(
            parents=True,
            exist_ok=True
        )

        (
            OUTPUT_DATASET /
            split /
            "labels"
        ).mkdir(
            parents=True,
            exist_ok=True
        )


# ============================================================
# CLEAN ONE LABEL FILE
# ============================================================

def clean_label_file(
    source_label,
    destination_label
):

    cleaned_lines = []

    if source_label.exists():

        with open(
            source_label,
            "r",
            encoding="utf-8"
        ) as file:

            for line in file:

                line = line.strip()

                if not line:
                    continue

                parts = line.split()

                if len(parts) < 5:
                    continue

                old_class_id = int(
                    float(parts[0])
                )

                # Ignore unwanted classes
                if old_class_id not in KEEP_CLASS_IDS:
                    continue

                # Remap kept class to class 0
                parts[0] = "0"

                cleaned_lines.append(
                    " ".join(parts)
                )


    # Create label file even when empty.
    #
    # Empty file = negative/background image.
    with open(
        destination_label,
        "w",
        encoding="utf-8"
    ) as file:

        if cleaned_lines:

            file.write(
                "\n".join(
                    cleaned_lines
                )
            )

            file.write("\n")


    return len(cleaned_lines)


# ============================================================
# PROCESS ONE SPLIT
# ============================================================

def process_split(split):

    source_images = (
        SOURCE_DATASET /
        split /
        "images"
    )

    source_labels = (
        SOURCE_DATASET /
        split /
        "labels"
    )

    output_images = (
        OUTPUT_DATASET /
        split /
        "images"
    )

    output_labels = (
        OUTPUT_DATASET /
        split /
        "labels"
    )


    if not source_images.exists():

        print(
            f"[SKIPPED] Missing folder: "
            f"{source_images}"
        )

        return


    image_count = 0
    annotation_count = 0
    positive_images = 0
    negative_images = 0


    for image_path in source_images.iterdir():

        if (
            not image_path.is_file()
            or image_path.suffix.lower()
            not in IMAGE_EXTENSIONS
        ):
            continue


        image_count += 1


        # ---------------------------------------------
        # Copy original image
        # ---------------------------------------------

        destination_image = (
            output_images /
            image_path.name
        )

        shutil.copy2(
            image_path,
            destination_image
        )


        # ---------------------------------------------
        # Corresponding YOLO label
        # ---------------------------------------------

        source_label = (
            source_labels /
            f"{image_path.stem}.txt"
        )

        destination_label = (
            output_labels /
            f"{image_path.stem}.txt"
        )


        kept = clean_label_file(
            source_label,
            destination_label
        )


        annotation_count += kept


        if kept > 0:

            positive_images += 1

        else:

            negative_images += 1


    print(
        f"\n{split.upper()}"
    )

    print(
        f"Images copied:       "
        f"{image_count}"
    )

    print(
        f"Positive images:     "
        f"{positive_images}"
    )

    print(
        f"Negative images:     "
        f"{negative_images}"
    )

    print(
        f"Lanyard annotations: "
        f"{annotation_count}"
    )


# ============================================================
# CREATE NEW data.yaml
# ============================================================

def create_yaml():

    data = {

        "train": "train/images",

        "val": "valid/images",

        "test": "test/images",

        "nc": 1,

        "names": [
            NEW_CLASS_NAME
        ]
    }


    yaml_path = (
        OUTPUT_DATASET /
        "data.yaml"
    )


    with open(
        yaml_path,
        "w",
        encoding="utf-8"
    ) as file:

        yaml.safe_dump(
            data,
            file,
            sort_keys=False
        )


    print(
        f"\nCreated:"
        f"\n{yaml_path.resolve()}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "===================================="
    )

    print(
        "Creating Clean Lanyard Dataset"
    )

    print(
        "===================================="
    )


    if not SOURCE_DATASET.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n"
            f"{SOURCE_DATASET.resolve()}"
        )


    create_folders()


    for split in SPLITS:

        process_split(
            split
        )


    create_yaml()


    print(
        "\n===================================="
    )

    print(
        "CLEAN DATASET CREATED"
    )

    print(
        "===================================="
    )

    print(
        f"\nOutput:\n"
        f"{OUTPUT_DATASET.resolve()}"
    )


if __name__ == "__main__":

    main()