from pathlib import Path
from ultralytics import YOLO

import tensorflow as tf
import cv2
import os


BASE_DIR = Path(__file__).resolve().parent


COLLAR_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "collar_saved_model"
)

LANYARD_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "lanyard_saved_model"
)

SHOES_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "shoes_saved_model"
)


model_collar = YOLO(
    str(COLLAR_MODEL_PATH),
    task="detect"
)

model_lanyard = YOLO(
    str(LANYARD_MODEL_PATH),
    task="detect"
)

model_shoes = YOLO(
    str(SHOES_MODEL_PATH),
    task="detect"
)


print("All SRD models loaded.")

# ==============================
# LABEL DEFINITIONS
# ==============================
COLLAR_LABEL = "collar"
LANYARD_LABELS = {"ID-CARD", "lanyard", "student_card"}
SHOES_LABELS = {"shoes", "shoe", "sneaker", "Sneaker", "Shoes"}
NO_SHOES_LABELS = {"no_shoes", "no-shoes", "barefoot", "NoShoes"}


# =====================================================
# STATIC IMAGE SRD (UPLOAD)
# =====================================================
def run_srd(image_path: str, output_path: str):
    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError("Failed to read image")

    annotated, data = run_srd_frame(frame)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, annotated)

    return data


# =====================================================
# REALTIME FRAME SRD (WEBCAM)
# =====================================================
def run_srd_frame(frame):
    H, W = frame.shape[:2]

    res_collar = model_collar(
        frame,
        conf=0.15,
        imgsz=640
    )

    res_lanyard = model_lanyard(
        frame,
        conf=0.15,
        imgsz=640
    )

    res_shoes = model_shoes(
        frame,
        conf=0.15,
        imgsz=640,
        iou=0.5
    )
    has_collar = False
    has_lanyard = False
    has_shoes = False

    # ---------- COLLAR ----------
    for box in res_collar[0].boxes:
        cls = res_collar[0].names[int(box.cls[0])]
        if cls != COLLAR_LABEL:
            continue
        has_collar = True
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, "collar", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # ---------- LANYARD ----------
    for box in res_lanyard[0].boxes:
        cls = res_lanyard[0].names[int(box.cls[0])]
        if cls not in LANYARD_LABELS:
            continue
        has_lanyard = True
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "lanyard", (x1, y2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # ---------- SHOES ----------
    for box in res_shoes[0].boxes:
        cls = res_shoes[0].names[int(box.cls[0])]
        if cls not in SHOES_LABELS and cls not in NO_SHOES_LABELS:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        bw, bh = x2 - x1, y2 - y1
        area = bw * bh

        if y2 < int(H * 0.55):
            continue
        if area > (W * H) * 0.12:
            continue
        if bw / max(bh, 1) > 3.5:
            continue

        if cls in SHOES_LABELS:
            has_shoes = True
            color = (0, 200, 0)
            label = "shoes"
        else:
            color = (0, 165, 255)
            label = "NO SHOES"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y2 + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # ---------- STATUS ----------
    compliant = has_collar and has_lanyard and has_shoes
    status_text = "STUDENT COMPLIANT" if compliant else "NOT COMPLIANT"
    status_color = (0, 255, 0) if compliant else (0, 0, 255)

    cv2.rectangle(frame, (20, 15), (560, 85), (0, 0, 0), -1)
    cv2.putText(frame, status_text, (30, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, status_color, 3)

    return frame, {
        "has_collar": has_collar,
        "has_lanyard": has_lanyard,
        "has_shoes": has_shoes,
        "compliant": compliant,
        "status_text": status_text
    }
