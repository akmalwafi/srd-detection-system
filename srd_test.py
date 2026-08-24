from ultralytics import YOLO
import cv2
import os

# ==============================
# LOAD MODELS (ONCE)
# ==============================
model_collar  = YOLO("runs/detect/train4/weights/best.pt")
model_lanyard = YOLO("runs/detect/train10/weights/best.pt")
model_shoes   = YOLO("runs/detect/train8/weights/best.pt")

# ==============================
# LABEL DEFINITIONS
# ==============================
COLLAR_LABEL = "collar"
LANYARD_LABELS = {"ID-CARD", "lanyard", "student_card"}
SHOES_LABELS = {"shoes", "shoe", "sneaker", "Sneaker", "Shoes"}
NO_SHOES_LABELS = {"no_shoes", "no-shoes", "barefoot", "NoShoes"}


def run_srd(image_path: str, output_path: str):
    """
    Runs SRD detection on an image.
    Saves annotated image to output_path.
    Returns result dictionary.
    """

    # Run models
    res_collar  = model_collar(image_path, conf=0.20, imgsz=1280)
    res_lanyard = model_lanyard(image_path, conf=0.30, imgsz=1280)
    res_shoes   = model_shoes(image_path, conf=0.40, imgsz=1280, iou=0.5)

    frame = cv2.imread(image_path)
    if frame is None:
        raise ValueError("Failed to read image")

    H, W = frame.shape[:2]

    has_collar = False
    has_lanyard = False
    has_shoes = False

    # ==============================
    # COLLAR
    # ==============================
    for box in res_collar[0].boxes:
        cls = res_collar[0].names[int(box.cls[0])]
        if cls != COLLAR_LABEL:
            continue

        has_collar = True
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, "collar", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # ==============================
    # LANYARD
    # ==============================
    for box in res_lanyard[0].boxes:
        cls = res_lanyard[0].names[int(box.cls[0])]
        if cls not in LANYARD_LABELS:
            continue

        has_lanyard = True
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "lanyard", (x1, y2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # ==============================
    # SHOES / NO SHOES
    # ==============================
    for box in res_shoes[0].boxes:
        cls = res_shoes[0].names[int(box.cls[0])]
        if cls not in SHOES_LABELS and cls not in NO_SHOES_LABELS:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        bw, bh = x2 - x1, y2 - y1
        area = bw * bh

        # bottom-region filter
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
        cv2.putText(frame, label, (x1, y2 + 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # ==============================
    # FINAL STATUS
    # ==============================
    compliant = has_collar and has_lanyard and has_shoes
    status_text = "STUDENT COMPLIANT" if compliant else "NOT COMPLIANT"
    status_color = (0, 255, 0) if compliant else (0, 0, 255)

    cv2.rectangle(frame, (20, 15), (560, 85), (0, 0, 0), -1)
    cv2.putText(frame, status_text, (30, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, status_color, 3)

    # ==============================
    # SAVE OUTPUT
    # ==============================
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, frame)

    return {
        "has_collar": has_collar,
        "has_lanyard": has_lanyard,
        "has_shoes": has_shoes,
        "compliant": compliant,
        "status_text": status_text
    }
