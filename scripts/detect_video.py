from ultralytics import YOLO
import cv2

# Load YOLO11 model
model = YOLO("yolo11n.pt")

# Open video
cap = cv2.VideoCapture("assets/traffic.mp4")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Detect objects
    results = model(frame)

    # Draw detections
    annotated_frame = results[0].plot()

    # Show video
    cv2.imshow("YOLO Video Detection", annotated_frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()