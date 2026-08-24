from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data=r"C:\Users\akmal\PycharmProjects\SRDDETECTION\formal and non formal shoes.v1i.yolov8\data.yaml",  # change to your real path
        epochs=100,
        imgsz=640,
        batch=16,
        device=0,      # GPU
        workers=8      # can adjust
    )

if __name__ == "__main__":
    main()
