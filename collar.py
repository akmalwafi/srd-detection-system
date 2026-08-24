from ultralytics import YOLO

# Load the pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")

# Start training
model.train(
    data="C:\\Users\\akmal\\PycharmProjects\\AIProject\\collar.v1i.yolov8\\data.yaml",  # Path to your Mask Merge dataset YAML
    epochs=70,  # Number of epochs
    imgsz=640  # Image size for training
)