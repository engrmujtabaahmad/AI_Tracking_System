import cv2
import os
import sqlite3
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_PATH = "assets/Traffic1.mp4"

OUTPUT_DIR = "plate_detector/outputs"
OUTPUT_VIDEO = os.path.join(
    OUTPUT_DIR,
    "person_tracking_output.mp4"
)

DATABASE_PATH = "plate_database/traffic_tracking.db"

CAMERA_ID = "Camera_01"
CITY = "Lahore"
ROAD = "Canal Road"

# Minimum YOLO confidence for detecting a person
PERSON_CONFIDENCE = 0.40


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("SMART TRAFFIC MONITORING SYSTEM")
print("PERSON DETECTION + TRACKING")
print("=" * 70)


# ============================================================
# LOAD YOLO MODEL
# ============================================================

print("\nLoading YOLO model...")

try:
    model = YOLO("yolo11n.pt")
    print("YOLO model loaded successfully.")
except Exception as e:
    print("\nERROR loading YOLO model:")
    print(e)
    exit()


# ============================================================
# DATABASE
# ============================================================

print("\nConnecting to tracking database...")

try:
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS person_tracking (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER,
            camera_id TEXT,
            city TEXT,
            road TEXT,
            timestamp TEXT,
            frame INTEGER,
            confidence REAL,
            video_source TEXT
        )
    """)

    conn.commit()

    print("Person tracking database ready.")

except Exception as e:
    print("\nERROR connecting to database:")
    print(e)
    exit()


# ============================================================
# OPEN VIDEO
# ============================================================

print("\n" + "=" * 70)
print("OPENING VIDEO")
print("=" * 70)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("\nERROR: Could not open video:")
    print(os.path.abspath(VIDEO_PATH))
    conn.close()
    exit()


# ============================================================
# VIDEO INFORMATION
# ============================================================

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if fps <= 0:
    fps = 24.0

duration = total_frames / fps


print("\n" + "=" * 70)
print("VIDEO INFORMATION")
print("=" * 70)

print(f"Video       : {VIDEO_PATH}")
print(f"FPS         : {fps:.2f}")
print(f"Resolution  : {width} x {height}")
print(f"Frames      : {total_frames}")
print(f"Duration    : {duration:.2f} seconds")
print(f"Camera ID   : {CAMERA_ID}")
print(f"City        : {CITY}")
print(f"Road        : {ROAD}")


# ============================================================
# OUTPUT VIDEO
# ============================================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width, height)
)


# ============================================================
# TRACKING
# ============================================================

print("\n" + "=" * 70)
print("STARTING PERSON DETECTION + TRACKING")
print("=" * 70)

frame_number = 0

detected_person_ids = set()
saved_events = set()

while True:

    success, frame = cap.read()

    if not success:
        break

    frame_number += 1

    # --------------------------------------------------------
    # YOLO TRACKING
    # --------------------------------------------------------

    results = model.track(
        frame,
        persist=True,
        classes=[0],              # COCO class 0 = person
        conf=PERSON_CONFIDENCE,
        verbose=False
    )

    current_person_ids = set()

    for result in results:

        if result.boxes is None:
            continue

        boxes = result.boxes

        for box in boxes:

            # ------------------------------------------------
            # CONFIDENCE
            # ------------------------------------------------

            confidence = float(box.conf[0])

            # ------------------------------------------------
            # TRACK ID
            # ------------------------------------------------

            if box.id is not None:
                person_id = int(box.id[0])
            else:
                person_id = -1

            current_person_ids.add(person_id)

            if person_id != -1:
                detected_person_ids.add(person_id)

            # ------------------------------------------------
            # BOUNDING BOX
            # ------------------------------------------------

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            # Keep coordinates inside frame
            x1 = max(0, min(x1, width - 1))
            y1 = max(0, min(y1, height - 1))
            x2 = max(0, min(x2, width - 1))
            y2 = max(0, min(y2, height - 1))

            # ------------------------------------------------
            # DRAW PERSON BOX
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # ------------------------------------------------
            # LABEL
            # ------------------------------------------------

            if person_id != -1:

                label = (
                    f"Person ID: {person_id} "
                    f"| Conf: {confidence:.2f}"
                )

            else:

                label = (
                    f"Person | "
                    f"Conf: {confidence:.2f}"
                )

            cv2.putText(
                frame,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

            # ------------------------------------------------
            # TIMESTAMP
            # ------------------------------------------------

            video_seconds = frame_number / fps

            hours = int(video_seconds // 3600)
            minutes = int((video_seconds % 3600) // 60)
            seconds = int(video_seconds % 60)

            timestamp = (
                f"{hours}:{minutes:02d}:{seconds:02d}"
            )

            # ------------------------------------------------
            # SAVE TRACKING EVENT
            #
            # Save only once per person per frame.
            # ------------------------------------------------

            event_key = (
                person_id,
                frame_number
            )

            if person_id != -1 and event_key not in saved_events:

                cursor.execute("""
                    INSERT INTO person_tracking (
                        person_id,
                        camera_id,
                        city,
                        road,
                        timestamp,
                        frame,
                        confidence,
                        video_source
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    person_id,
                    CAMERA_ID,
                    CITY,
                    ROAD,
                    timestamp,
                    frame_number,
                    confidence,
                    os.path.basename(VIDEO_PATH)
                ))

                saved_events.add(event_key)

    # --------------------------------------------------------
    # DISPLAY INFORMATION ON VIDEO
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Camera: {CAMERA_ID}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Frame: {frame_number}/{total_frames}",
        (20, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Persons Tracked: {len(detected_person_ids)}",
        (20, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # --------------------------------------------------------
    # SAVE OUTPUT FRAME
    # --------------------------------------------------------

    out.write(frame)

    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if frame_number % 20 == 0:

        progress = (
            frame_number / total_frames
        ) * 100

        print(
            f"Processing: "
            f"{progress:.1f}% "
            f"({frame_number}/{total_frames})"
        )


# ============================================================
# CLEANUP
# ============================================================

cap.release()
out.release()

conn.commit()
conn.close()


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 70)
print("PERSON TRACKING COMPLETED")
print("=" * 70)

print(f"Unique persons tracked : {len(detected_person_ids)}")

print("\nTracked Person IDs:")

if detected_person_ids:

    for person_id in sorted(detected_person_ids):
        print(f"  ✓ Person_{person_id}")

else:

    print("  No persons detected.")


print("\n" + "=" * 70)
print("OUTPUT FILES")
print("=" * 70)

print(f"Output Video:")
print(os.path.abspath(OUTPUT_VIDEO))

print("\nDatabase:")
print(os.path.abspath(DATABASE_PATH))

print("=" * 70)

print("\nProcessing finished successfully.")