from pathlib import Path
import cv2


DATASET = Path("Lanyard3.v3i.yolov8")

IMAGES_DIR = DATASET / "train" / "images"
LABELS_DIR = DATASET / "train" / "labels"

OUTPUT_DIR = Path("lanyard_class_check")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# We only want to inspect these
TARGET_CLASSES = {
    3,
    5
}

MAX_IMAGES = 30


image_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
}


count = 0


for image_path in IMAGES_DIR.iterdir():

    if image_path.suffix.lower() not in image_extensions:
        continue

    label_path = (
        LABELS_DIR /
        f"{image_path.stem}.txt"
    )

    if not label_path.exists():
        continue


    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        continue


    h, w = image.shape[:2]

    found_target = False


    with open(
        label_path,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            values = line.strip().split()

            if len(values) < 5:
                continue


            class_id = int(
                float(values[0])
            )


            if class_id not in TARGET_CLASSES:
                continue


            found_target = True


            xc = float(values[1]) * w
            yc = float(values[2]) * h

            bw = float(values[3]) * w
            bh = float(values[4]) * h


            x1 = int(
                xc - bw / 2
            )

            y1 = int(
                yc - bh / 2
            )

            x2 = int(
                xc + bw / 2
            )

            y2 = int(
                yc + bh / 2
            )


            if class_id == 3:
                label = "CLASS 3"

            else:
                label = "CLASS 5 / ID-CARD"


            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                3
            )


            cv2.putText(
                image,
                label,
                (
                    x1,
                    max(30, y1 - 10)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )


    if found_target:

        output_path = (
            OUTPUT_DIR /
            image_path.name
        )

        cv2.imwrite(
            str(output_path),
            image
        )

        count += 1


        if count >= MAX_IMAGES:
            break


print(
    f"Saved {count} images to:"
)

print(
    OUTPUT_DIR.resolve()
)