from ultralytics import YOLO
import cv2

# Load ONLY hijab model
model = YOLO("runs/detect/train11/weights/best.pt")

# Image path
image_path = "test_images/hijab.jpg"

# Run detection
results = model(image_path, conf=0.5)

# Draw detections
annotated = results[0].plot()

# ---- Extract detected classes safely ----
detected_items = set()

if results[0].boxes is not None:
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        class_name = results[0].names[cls_id]
        detected_items.add(class_name)

print("Detected items:", detected_items)

# Simple hijab status
if "hijab" in detected_items:
    status = "HIJAB DETECTED"
else:
    status = "NO HIJAB DETECTED"

print("Status:", status)

# Show result
cv2.imshow("Hijab Detection Result", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
