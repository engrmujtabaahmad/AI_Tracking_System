from ultralytics import YOLO
import cv2
import os
import csv
from datetime import datetime

# ============================
# Configuration
# ============================

VIDEO_PATH = "assets/Traffic1.mp4"

OUTPUT_FOLDER = "vehicle_database"

CSV_PATH = "database/vehicle_metadata.csv"

CAMERA_ID = "Camera_01"

CITY = "Lahore"

ROAD = "Canal Road"

# Vehicle Classes
VEHICLE_CLASSES = [
    "car",
    "motorcycle",
    "bus",
    "truck"
]

# ============================
# Create folders
# ============================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ============================
# Load YOLO11
# ============================

model = YOLO("yolo11n.pt")

# ============================
# Open Video
# ============================

cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)

frame_number = 0

saved_ids = set()

# ============================
# Create CSV
# ============================

with open(CSV_PATH, "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "Image",
        "CameraID",
        "Location",
        "Road",
        "Timestamp",
        "Frame",
        "VehicleType"
    ])

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False
        )

        result = results[0]

        if result.boxes.id is None:
            continue

        boxes = result.boxes.xyxy.cpu().numpy()

        ids = result.boxes.id.cpu().numpy().astype(int)

        classes = result.boxes.cls.cpu().numpy().astype(int)

        for box, track_id, cls in zip(boxes, ids, classes):

            class_name = model.names[cls]

            if class_name not in VEHICLE_CLASSES:
                continue

            if track_id in saved_ids:
                continue

            saved_ids.add(track_id)

            x1, y1, x2, y2 = map(int, box)

            crop = frame[y1:y2, x1:x2]

            filename = f"{class_name}_{track_id}.jpg"

            save_path = os.path.join(
                OUTPUT_FOLDER,
                filename
            )

            cv2.imwrite(save_path, crop)

            seconds = frame_number / fps

            timestamp = str(datetime.fromtimestamp(seconds).time())

            writer.writerow([
                filename,
                CAMERA_ID,
                CITY,
                ROAD,
                timestamp,
                frame_number,
                class_name.capitalize()
            ])

            print(f"Saved {filename}")

cap.release()

print("\nFinished Successfully!")