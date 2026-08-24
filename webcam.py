from ultralytics import YOLO
import cv2

# ==============================
# LOAD 3 MODELS
# ==============================
model_collar  = YOLO(r"runs/detect/train4/weights/best.pt")
model_lanyard = YOLO(r"runs/detect/train7/weights/best.pt")
model_shoes   = YOLO(r"runs/detect/train8/weights/best.pt")   # shoes/no_shoes

# ==============================
# LABEL FILTERS (case-sensitive)
# ==============================
COLLAR_LABEL = "collar"
LANYARD_LABELS = {"ID-CARD", "lanyard", "student_card"}  # add/remove based on your model
SHOES_LABELS = {"shoes", "shoe", "sneaker", "Sneaker", "Shoes"}
NO_SHOES_LABELS = {"no_shoes", "no-shoes", "barefoot", "NoShoes"}

# ==============================
# WEBCAM SETUP
# ==============================
cap = cv2.VideoCapture(0)  # change to 1 if external webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    H, W = frame.shape[:2]

    # Run models on current frame (same settings as your image code)
    res_collar  = model_collar(frame,  conf=0.50, imgsz=1280, verbose=False)
    res_lanyard = model_lanyard(frame, conf=0.50, imgsz=1280, verbose=False)
    res_shoes   = model_shoes(frame,   conf=0.50, imgsz=1280, iou=0.5, verbose=False)

    # ==============================
    # DRAW COLLAR (BLUE)
    # ==============================
    for box in res_collar[0].boxes:
        cls_name = res_collar[0].names[int(box.cls[0])]
        if cls_name != COLLAR_LABEL:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, f"collar {conf:.2f}", (x1, y1 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # ==============================
    # DRAW LANYARD (GREEN)
    # ==============================
    for box in res_lanyard[0].boxes:
        cls_name = res_lanyard[0].names[int(box.cls[0])]
        if cls_name not in LANYARD_LABELS:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"lanyard {conf:.2f}", (x1, y2 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # ==============================
    # DRAW SHOES / NO SHOES (with same car filter)
    # ==============================
    for box in res_shoes[0].boxes:
        cls_name = res_shoes[0].names[int(box.cls[0])]
        conf = float(box.conf[0])

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        bw = x2 - x1
        bh = y2 - y1
        area = bw * bh

        if cls_name not in SHOES_LABELS and cls_name not in NO_SHOES_LABELS:
            continue

        # same filter rules you used
        if y2 < int(H * 0.55):
            continue
        if area > (W * H) * 0.12:
            continue
        aspect = bw / max(bh, 1)
        if aspect > 3.5:
            continue

        if cls_name in SHOES_LABELS:
            color = (0, 200, 0)
            label = f"shoes {conf:.2f}"
        else:
            color = (0, 165, 255)
            label = f"NO SHOES {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y2 + 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Show webcam
    cv2.imshow("Webcam - Collar + Lanyard + Shoes", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
        break

cap.release()
cv2.destroyAllWindows()
