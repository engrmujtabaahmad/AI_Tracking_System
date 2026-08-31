from ultralytics import YOLO

# Load YOLO11 Nano model
model = YOLO("yolo11n.pt")

# Start training
model.train(
    data="datasets/license_plate/data.yaml",
    epochs=20,
    imgsz=640,
    batch=4,
    workers=0,

    
    device="cpu",
    project="trained_models",
    name="license_plate_yolo11"
)

print("\nTraining Completed Successfully!")