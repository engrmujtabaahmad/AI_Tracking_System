from ultralytics import YOLO
import cv2
import os

# ==========================================
# Load trained license plate model
# ==========================================

model_path = "plate_detector/models/best.pt"

model = YOLO(model_path)

# ==========================================
# Input image
# ==========================================

image_path = "plate_detector/test_images/test_car.jpg"

# ==========================================
# Output location
# ==========================================

output_dir = "plate_detector/outputs"
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, "detected_plate.jpg")

# ==========================================
# Run detection
# ==========================================

results = model.predict(
    source=image_path,
    conf=0.40,
    imgsz=640
)

# ==========================================
# Get first result
# ==========================================

result = results[0]

# ==========================================
# Print detections
# ==========================================

print("=" * 50)
print("LICENSE PLATE DETECTION")
print("=" * 50)

if result.boxes is None or len(result.boxes) == 0:

    print("No license plate detected.")

else:

    for i, box in enumerate(result.boxes):

        confidence = float(box.conf[0])

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )

        print(
            f"Plate {i + 1}: "
            f"Confidence = {confidence:.2f}, "
            f"Coordinates = ({x1}, {y1}, {x2}, {y2})"
        )

# ==========================================
# Draw detections
# ==========================================

annotated_image = result.plot()

# ==========================================
# Save result
# ==========================================

cv2.imwrite(
    output_path,
    annotated_image
)

print("=" * 50)
print(f"Result saved to: {output_path}")
print("=" * 50)