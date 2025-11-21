from ultralytics import YOLO

# Load YOLO classification model (pretrained)
model = YOLO("yolov8n-cls.pt")

# Train on your dataset
model.train(
    data="data/classification",   # path to your dataset folder
    epochs=30,                    # increase if needed
    imgsz=224,                    # default for classification
    batch=16
)
