from ultralytics import YOLO
import cv2
from collections import Counter

# Load YOLO model
model = YOLO("yolo11n.pt")

# Load image
image = cv2.imread("assets/traffic.jfif")

# Run detection
results = model(image)

# Class names
names = model.names

# Counter
counter = Counter()

# Process detections
for result in results:
    for box in result.boxes:
        cls = int(box.cls[0])
        class_name = names[cls]
        counter[class_name] += 1

# Print Report
print("=" * 40)
print("Traffic Analysis Report")
print("=" * 40)

for obj, count in counter.items():
    print(f"{obj:<15}: {count}")

# Calculate Traffic Density
vehicle_count = (
    counter["car"] +
    counter["bus"] +
    counter["truck"] +
    counter["motorcycle"]
)

print("\nTotal Vehicles:", vehicle_count)

if vehicle_count < 10:
    density = "LOW"
elif vehicle_count < 25:
    density = "MEDIUM"
elif vehicle_count < 50:
    density = "HIGH"
else:
    density = "VERY HIGH"

print("Traffic Density:", density)

print("=" * 40)

# Show image
annotated = results[0].plot()

cv2.imshow("Traffic Analysis", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()