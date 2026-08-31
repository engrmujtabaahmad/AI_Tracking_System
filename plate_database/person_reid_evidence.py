import os
import sqlite3
import cv2
from ultralytics import YOLO


# ============================================================
# SMART TRAFFIC MONITORING SYSTEM
# PERSON RE-ID EVIDENCE GENERATOR
# ============================================================


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DB_PATH = os.path.join(
    BASE_DIR,
    "plate_database",
    "traffic_tracking.db"
)

ASSETS_DIR = os.path.join(
    BASE_DIR,
    "assets"
)

EVIDENCE_DIR = os.path.join(
    ASSETS_DIR,
    "reid_evidence"
)

os.makedirs(
    EVIDENCE_DIR,
    exist_ok=True
)


# ============================================================
# CAMERA VIDEOS
# ============================================================

CAMERA_VIDEOS = {

    "Unknown_Camera": {
        "video": os.path.join(
            ASSETS_DIR,
            "traffic.mp4"
        )
    },

    "Camera_01": {
        "video": os.path.join(
            ASSETS_DIR,
            "Traffic1.mp4"
        )
    },

    "Camera_02": {
        "video": os.path.join(
            ASSETS_DIR,
            "Traffic2.mp4"
        )
    }
}


# ============================================================
# DATABASE
# ============================================================

def get_connection():

    return sqlite3.connect(
        DB_PATH
    )


# ============================================================
# ADD EVIDENCE COLUMNS IF NECESSARY
# ============================================================

def prepare_database():

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute(
        "PRAGMA table_info(person_reid_sightings)"
    )

    columns = [
        row[1]
        for row in cursor.fetchall()
    ]

    if "evidence_frame_path" not in columns:

        cursor.execute(
            """
            ALTER TABLE person_reid_sightings
            ADD COLUMN evidence_frame_path TEXT
            """
        )

    if "evidence_crop_path" not in columns:

        cursor.execute(
            """
            ALTER TABLE person_reid_sightings
            ADD COLUMN evidence_crop_path TEXT
            """
        )

    conn.commit()

    conn.close()


# ============================================================
# GET BEST MATCH
# ============================================================

def get_best_match():

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            id,
            reference_image,
            camera_id,
            video_source,
            first_seen_frame,
            last_seen_frame,
            best_similarity,
            best_yolo_confidence,
            matching_frames
        FROM person_reid_sightings
        ORDER BY best_similarity DESC
        LIMIT 1
        """
    )

    result = cursor.fetchone()

    conn.close()

    return result


# ============================================================
# SAVE EVIDENCE PATHS
# ============================================================

def update_database(
    record_id,
    frame_path,
    crop_path
):

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute(
        """
        UPDATE person_reid_sightings
        SET
            evidence_frame_path = ?,
            evidence_crop_path = ?
        WHERE id = ?
        """,
        (
            frame_path,
            crop_path,
            record_id
        )
    )

    conn.commit()

    conn.close()


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("SMART TRAFFIC MONITORING SYSTEM")
print("PERSON RE-ID EVIDENCE GENERATOR")
print("=" * 70)

print()


# ============================================================
# PREPARE DATABASE
# ============================================================

print("Preparing database...")

prepare_database()

print("Database ready.")

print()


# ============================================================
# GET MATCH
# ============================================================

match = get_best_match()

if match is None:

    print(
        "ERROR: No Person Re-ID result found."
    )

    print(
        "Run person_reid.py first."
    )

    raise SystemExit


(
    record_id,
    reference_image,
    camera_id,
    video_source,
    first_seen_frame,
    last_seen_frame,
    best_similarity,
    best_yolo_confidence,
    matching_frames
) = match


print("Best Re-ID result found.")

print(
    "Reference image:",
    reference_image
)

print(
    "Camera:",
    camera_id
)

print(
    "Video:",
    video_source
)

print(
    "Best similarity:",
    round(
        best_similarity,
        3
    )
)

print(
    "Best frame range:",
    first_seen_frame,
    "-",
    last_seen_frame
)

print()


# ============================================================
# VIDEO PATH
# ============================================================

camera_info = CAMERA_VIDEOS.get(
    camera_id
)

if camera_info is None:

    print(
        "ERROR: Camera not configured."
    )

    raise SystemExit


video_path = camera_info["video"]


if not os.path.exists(video_path):

    print(
        "ERROR: Video not found:"
    )

    print(video_path)

    raise SystemExit


# ============================================================
# LOAD YOLO
# ============================================================

print("Loading YOLO model...")

try:

    yolo_model = YOLO(
        "yolo11n.pt"
    )

except Exception as e:

    print(
        "ERROR loading YOLO:"
    )

    print(e)

    raise SystemExit


print(
    "YOLO model loaded."
)

print()


# ============================================================
# OPEN VIDEO
# ============================================================

print(
    "Opening video:",
    video_path
)

cap = cv2.VideoCapture(
    video_path
)

if not cap.isOpened():

    print(
        "ERROR: Could not open video."
    )

    raise SystemExit


fps = cap.get(
    cv2.CAP_PROP_FPS
)

if fps <= 0:

    fps = 25.0


# ============================================================
# BEST FRAME
# ============================================================

best_frame_number = int(
    first_seen_frame
)


# Seek to frame

cap.set(
    cv2.CAP_PROP_POS_FRAMES,
    best_frame_number - 1
)


ret, frame = cap.read()


if not ret:

    print(
        "ERROR: Could not read evidence frame."
    )

    cap.release()

    raise SystemExit


print(
    "Evidence frame loaded."
)

print(
    "Frame:",
    best_frame_number
)

print()


# ============================================================
# DETECT PERSONS
# ============================================================

print(
    "Detecting persons in evidence frame..."
)

results = yolo_model(
    frame,
    classes=[0],
    verbose=False
)


person_boxes = []


for result in results:

    if result.boxes is None:

        continue

    for box in result.boxes:

        confidence = float(
            box.conf[0]
        )

        if confidence < 0.40:

            continue

        coordinates = (
            box.xyxy[0]
            .cpu()
            .numpy()
        )

        person_boxes.append(
            (
                coordinates,
                confidence
            )
        )


if not person_boxes:

    print(
        "WARNING: No person detected "
        "in evidence frame."
    )

    # Still save original frame

    evidence_frame_name = (
        f"{camera_id}_"
        f"frame_{best_frame_number}_"
        f"similarity_{best_similarity:.3f}.jpg"
    )

    evidence_frame_path = os.path.join(
        EVIDENCE_DIR,
        evidence_frame_name
    )

    cv2.imwrite(
        evidence_frame_path,
        frame
    )

    update_database(
        record_id,
        evidence_frame_path,
        ""
    )

    cap.release()

    print()
    print(
        "Evidence frame saved:"
    )

    print(
        evidence_frame_path
    )

    raise SystemExit


# ============================================================
# SELECT BEST PERSON
# ============================================================

best_box, best_confidence = max(
    person_boxes,
    key=lambda x: x[1]
)


x1, y1, x2, y2 = map(
    int,
    best_box
)


height, width = frame.shape[:2]


x1 = max(
    0,
    x1
)

y1 = max(
    0,
    y1
)

x2 = min(
    width,
    x2
)

y2 = min(
    height,
    y2
)


# ============================================================
# CROP PERSON
# ============================================================

person_crop = frame[
    y1:y2,
    x1:x2
]


if person_crop.size == 0:

    print(
        "ERROR: Invalid person crop."
    )

    cap.release()

    raise SystemExit


# ============================================================
# DRAW EVIDENCE BOX
# ============================================================

evidence_frame = frame.copy()


cv2.rectangle(
    evidence_frame,
    (x1, y1),
    (x2, y2),
    (0, 255, 0),
    4
)


label = (
    f"PERSON MATCH | "
    f"Similarity: {best_similarity:.3f}"
)


cv2.rectangle(
    evidence_frame,
    (x1, max(0, y1 - 45)),
    (x1 + 600, y1),
    (0, 255, 0),
    -1
)


cv2.putText(
    evidence_frame,
    label,
    (x1 + 10, max(30, y1 - 12)),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.9,
    (0, 0, 0),
    2
)


# ============================================================
# SAVE FRAME
# ============================================================

evidence_frame_name = (
    f"{camera_id}_"
    f"frame_{best_frame_number}_"
    f"similarity_{best_similarity:.3f}.jpg"
)


evidence_frame_path = os.path.join(
    EVIDENCE_DIR,
    evidence_frame_name
)


cv2.imwrite(
    evidence_frame_path,
    evidence_frame
)


# ============================================================
# SAVE PERSON CROP
# ============================================================

evidence_crop_name = (
    f"{camera_id}_"
    f"person_frame_{best_frame_number}.jpg"
)


evidence_crop_path = os.path.join(
    EVIDENCE_DIR,
    evidence_crop_name
)


cv2.imwrite(
    evidence_crop_path,
    person_crop
)


# ============================================================
# UPDATE DATABASE
# ============================================================

update_database(
    record_id,
    evidence_frame_path,
    evidence_crop_path
)


# ============================================================
# CLOSE VIDEO
# ============================================================

cap.release()


# ============================================================
# RESULT
# ============================================================

print()
print("=" * 70)

print(
    "PERSON RE-ID EVIDENCE GENERATED"
)

print("=" * 70)

print()

print(
    "Camera:",
    camera_id
)

print(
    "Frame:",
    best_frame_number
)

print(
    "Similarity:",
    round(
        best_similarity,
        3
    )
)

print(
    "YOLO Confidence:",
    round(
        best_confidence,
        3
    )
)

print()

print(
    "Evidence frame:"
)

print(
    evidence_frame_path
)

print()

print(
    "Person crop:"
)

print(
    evidence_crop_path
)

print()

print(
    "Database updated successfully."
)

print()
print("=" * 70)