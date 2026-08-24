from pathlib import Path
from collections import Counter

DATASET = Path("Lanyard3.v3i.yolov8")

for split in ["train", "valid", "test"]:
    labels_dir = DATASET / split / "labels"

    counts = Counter()
    files = 0

    for label_file in labels_dir.glob("*.txt"):
        files += 1

        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()

                if len(parts) >= 5:
                    class_id = int(float(parts[0]))
                    counts[class_id] += 1

    print(f"\n{split.upper()}")
    print("Label files:", files)

    for class_id, count in sorted(counts.items()):
        print(
            f"class {class_id}: "
            f"{count} annotations"
        )