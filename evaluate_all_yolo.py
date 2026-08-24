from ultralytics import YOLO
from pathlib import Path
import csv
import numpy as np


# ============================================================
# MODEL + DATASET CONFIGURATION
# ============================================================

MODELS = {
    "collar": {
        "model": Path(
            r"runs\detect\train4\weights\best.pt"
        ),
        "data": Path(
            r"collar.v1i.yolov8\data.yaml"
        ),
    },

    "lanyard": {
        "model": Path(
            r"runs\detect\train7\weights\best.pt"
        ),
        "data": Path(
            r"Lanyard2.v1i.yolov8\data.yaml"
        ),
    },

    "shoes": {
        "model": Path(
            r"runs\detect\train8\weights\best.pt"
        ),
        "data": Path(
            r"formal and non formal shoes.v1i.yolov8\data.yaml"
        ),
    },
}


OUTPUT_ROOT = Path(
    "runs_evaluation"
)

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# MAIN
# ============================================================

def main():

    summary_rows = []

    print(
        "\n=============================================="
    )

    print(
        "SRD YOLO MODEL EVALUATION"
    )

    print(
        "=============================================="
    )


    for model_name, config in MODELS.items():

        model_path = config["model"]
        data_path = config["data"]


        print(
            f"\n\n=============================================="
        )

        print(
            f"EVALUATING: {model_name.upper()}"
        )

        print(
            "=============================================="
        )

        print(
            f"Model:   {model_path}"
        )

        print(
            f"Dataset: {data_path}"
        )


        # ----------------------------------------------------
        # Check paths
        # ----------------------------------------------------

        if not model_path.exists():

            print(
                f"[ERROR] Model not found:\n"
                f"{model_path.resolve()}"
            )

            continue


        if not data_path.exists():

            print(
                f"[ERROR] data.yaml not found:\n"
                f"{data_path.resolve()}"
            )

            continue


        # ----------------------------------------------------
        # Load YOLO model
        # ----------------------------------------------------

        model = YOLO(
            str(model_path)
        )


        # ----------------------------------------------------
        # Validate
        #
        # split="test":
        # use test/images defined in data.yaml
        #
        # plots=True:
        # generate confusion matrix + curves
        # ----------------------------------------------------

        metrics = model.val(
            data=str(data_path),
            split="test",
            imgsz=640,
            batch=4,
            plots=True,

            project=str(
                OUTPUT_ROOT
            ),

            name=model_name,

            exist_ok=True
        )


        # ====================================================
        # OVERALL METRICS
        # ====================================================

        precision = float(
            metrics.box.mp
        )

        recall = float(
            metrics.box.mr
        )

        map50 = float(
            metrics.box.map50
        )

        map50_95 = float(
            metrics.box.map
        )


        # Ultralytics also stores class-level F1 values.
        # Mean them for an overall F1.
        if len(metrics.box.f1) > 0:

            f1 = float(
                np.mean(
                    metrics.box.f1
                )
            )

        else:

            f1 = 0.0


        # ====================================================
        # PRINT RESULTS
        # ====================================================

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


        # ====================================================
        # SPEED
        # ====================================================

        print(
            "\nSpeed:"
        )

        print(
            metrics.speed
        )


        # ====================================================
        # PER-CLASS RESULTS
        # ====================================================

        print(
            "\nPER-CLASS RESULTS"
        )

        print(
            "----------------------------------------------"
        )


        for i, class_index in enumerate(
            metrics.box.ap_class_index
        ):

            class_index = int(
                class_index
            )

            class_name = (
                model.names[
                    class_index
                ]
            )


            class_precision = float(
                metrics.box.p[i]
            )

            class_recall = float(
                metrics.box.r[i]
            )

            class_f1 = float(
                metrics.box.f1[i]
            )

            class_map50 = float(
                metrics.box.ap50[i]
            )

            class_map50_95 = float(
                metrics.box.ap[i]
            )


            print(
                f"\nClass: {class_name}"
            )

            print(
                f"  Precision: "
                f"{class_precision:.4f}"
            )

            print(
                f"  Recall:    "
                f"{class_recall:.4f}"
            )

            print(
                f"  F1:        "
                f"{class_f1:.4f}"
            )

            print(
                f"  mAP50:     "
                f"{class_map50:.4f}"
            )

            print(
                f"  mAP50-95:  "
                f"{class_map50_95:.4f}"
            )


        # ====================================================
        # CONFUSION MATRIX
        # ====================================================

        print(
            "\nConfusion Matrix:"
        )

        try:

            confusion_df = (
                metrics
                .confusion_matrix
                .to_df()
            )

            print(
                confusion_df
            )

        except Exception as error:

            print(
                "Confusion matrix image was generated, "
                "but table export is not supported by "
                "your current Ultralytics version."
            )

            print(
                f"Details: {error}"
            )


        # ====================================================
        # SUMMARY
        # ====================================================

        summary_rows.append(
            {
                "Model": model_name,

                "Precision": precision,

                "Recall": recall,

                "F1": f1,

                "mAP50": map50,

                "mAP50-95": map50_95,

                "Inference_ms": (
                    metrics.speed.get(
                        "inference",
                        0
                    )
                )
            }
        )


    # ========================================================
    # SAVE SUMMARY CSV
    # ========================================================

    summary_path = (
        OUTPUT_ROOT /
        "srd_yolo_metrics.csv"
    )


    if summary_rows:

        with open(
            summary_path,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=(
                    summary_rows[0]
                    .keys()
                )
            )

            writer.writeheader()

            writer.writerows(
                summary_rows
            )


        print(
            "\n\n=============================================="
        )

        print(
            "FINAL SRD COMPARISON"
        )

        print(
            "=============================================="
        )


        print(
            "\n"
            f"{'Model':<12}"
            f"{'Precision':>12}"
            f"{'Recall':>12}"
            f"{'F1':>12}"
            f"{'mAP50':>12}"
            f"{'mAP50-95':>14}"
        )


        for row in summary_rows:

            print(
                f"{row['Model']:<12}"
                f"{row['Precision']:>12.4f}"
                f"{row['Recall']:>12.4f}"
                f"{row['F1']:>12.4f}"
                f"{row['mAP50']:>12.4f}"
                f"{row['mAP50-95']:>14.4f}"
            )


        print(
            f"\nSummary saved to:\n"
            f"{summary_path.resolve()}"
        )


if __name__ == "__main__":
    main()