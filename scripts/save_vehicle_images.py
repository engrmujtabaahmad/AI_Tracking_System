from ultralytics import YOLO
import cv2
import os

# Load YOLO model
model = YOLO("yolo11n.pt")

# Create output folder
os.makedirs("vehicle_database", exist_ok=True)

# Open video
cap = cv2.VideoCapture("assets/Traffic1.mp4")

saved_ids = set()

while True:

    success, frame = cap.read()

    if not success:
        break

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        conf=0.5,
        verbose=False
    )

    result = results[0]

    if result.boxes.id is not None:

        boxes = result.boxes.xyxy.cpu().numpy()
        ids = result.boxes.id.cpu().numpy().astype(int)
        classes = result.boxes.cls.cpu().numpy().astype(int)

        for box, track_id, cls in zip(boxes, ids, classes):

            class_name = model.names[cls]

            # Save only vehicles
            if class_name not in ["car", "bus", "truck", "motorcycle"]:
                continue

            if track_id in saved_ids:
                continue

            x1, y1, x2, y2 = map(int, box)

            crop = frame[y1:y2, x1:x2]

            filename = f"vehicle_database/{class_name}_{track_id}.jpg"

            cv2.imwrite(filename, crop)

            saved_ids.add(track_id)

            print(f"Saved: {filename}")

    annotated = result.plot()

    cv2.imshow("Vehicle Saving System", annotated)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("\nVehicle extraction completed.")