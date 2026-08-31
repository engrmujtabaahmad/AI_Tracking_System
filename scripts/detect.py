from ultralytics import YOLO
import cv2

# Load YOLO11 Nano model
model = YOLO("yolo11n.pt")

# Read image
image = cv2.imread("assets/traffic.jfif")

# Detect objects
results = model(image)

# Show detections
annotated = results[0].plot()

cv2.imshow("Detection", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()