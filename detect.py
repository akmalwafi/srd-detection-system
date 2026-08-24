from ultralytics import YOLO
import cv2

model = YOLO("runs/detect/train11/weights/best.pt")

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.5)
    annotated = results[0].plot()

    cv2.imshow("Hijab Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()