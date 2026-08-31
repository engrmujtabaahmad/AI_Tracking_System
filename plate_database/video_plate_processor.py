import cv2
import os
import re
import sqlite3
import easyocr

from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_PATH = "assets/Traffic1.mp4"

MODEL_PATH = "plate_detector/models/best.pt"

DATABASE_PATH = "plate_database/traffic_tracking.db"

OUTPUT_DIR = "plate_detector/outputs"

OUTPUT_VIDEO = os.path.join(
    OUTPUT_DIR,
    "Traffic1_database_tracking.mp4"
)

CAMERA_ID = "Camera_01"
CITY = "Lahore"
ROAD = "Canal Road"


# ============================================================
# SETTINGS
# ============================================================

YOLO_CONFIDENCE = 0.50

OCR_CONFIDENCE = 0.40

# Number of consistent observations required
# before saving a plate to the database.
MIN_OBSERVATIONS = 1

# Do not save the same plate repeatedly
# within this many frames.
SAVE_COOLDOWN_FRAMES = 48


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# NORMALIZE OCR TEXT
# ============================================================

def normalize_plate(text):

    if not text:
        return ""

    text = text.upper()

    # Remove everything except letters and numbers
    text = re.sub(r"[^A-Z0-9]", "", text)

    return text


# ============================================================
# VALIDATE PLATE
# ============================================================

def valid_plate(text):

    if not text:
        return False

    # Ignore extremely short OCR results
    if len(text) < 4:
        return False

    # Ignore extremely long OCR results
    if len(text) > 10:
        return False

    # Must contain at least one letter
    # and one number for a basic plate format.
    if not re.search(r"[A-Z]", text):
        return False

    if not re.search(r"[0-9]", text):
        return False

    return True


# ============================================================
# DATABASE SETUP
# ============================================================

def setup_database():

    os.makedirs("plate_database", exist_ok=True)

    connection = sqlite3.connect(DATABASE_PATH)

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS vehicle_detections (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            plate_number TEXT NOT NULL,

            camera_id TEXT NOT NULL,

            city TEXT NOT NULL,

            road TEXT NOT NULL,

            timestamp TEXT NOT NULL,

            frame INTEGER,

            confidence REAL,

            video_source TEXT,

            created_at TEXT DEFAULT CURRENT_TIMESTAMP

        )
    """)

    connection.commit()

    connection.close()


# ============================================================
# SAVE DATABASE RECORD
# ============================================================

def save_to_database(
    plate_number,
    camera_id,
    city,
    road,
    timestamp,
    frame,
    confidence,
    video_source
):

    connection = sqlite3.connect(DATABASE_PATH)

    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO vehicle_detections
        (
            plate_number,
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

        plate_number,
        camera_id,
        city,
        road,
        timestamp,
        frame,
        confidence,
        video_source

    ))

    connection.commit()

    connection.close()


# ============================================================
# LOAD YOLO
# ============================================================

print("=" * 60)
print("LOADING LICENSE PLATE DETECTOR")
print("=" * 60)

model = YOLO(MODEL_PATH)

print("YOLO11 license plate model loaded successfully.")


# ============================================================
# LOAD OCR
# ============================================================

print()
print("=" * 60)
print("LOADING OCR")
print("=" * 60)

reader = easyocr.Reader(
    ["en"],
    gpu=False
)

print("OCR loaded successfully.")


# ============================================================
# DATABASE
# ============================================================

setup_database()

print()
print("Tracking database ready.")


# ============================================================
# OPEN VIDEO
# ============================================================

print()
print("=" * 60)
print("OPENING VIDEO")
print("=" * 60)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():

    print()
    print("ERROR: Could not open video.")
    print(f"Video: {VIDEO_PATH}")

    raise SystemExit


# ============================================================
# VIDEO INFORMATION
# ============================================================

fps = cap.get(cv2.CAP_PROP_FPS)

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

duration = (
    total_frames / fps
    if fps > 0
    else 0
)


print()
print("=" * 60)
print("VIDEO INFORMATION")
print("=" * 60)

print(f"Video       : {VIDEO_PATH}")
print(f"FPS         : {fps:.2f}")
print(f"Resolution  : {width} x {height}")
print(f"Frames      : {total_frames}")
print(f"Duration    : {duration:.2f} seconds")

print(f"Camera ID   : {CAMERA_ID}")
print(f"City        : {CITY}")
print(f"Road        : {ROAD}")


# ============================================================
# VIDEO WRITER
# ============================================================

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

writer = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width, height)
)


# ============================================================
# TRACKING VARIABLES
# ============================================================

# Stores OCR observations.
#
# Example:
#
# {
#     "NI2022": [
#         0.61,
#         0.72,
#         0.81
#     ]
# }

observations = {}


# Last frame where a plate was saved
last_saved_frame = {}


# Keep summary information
confirmed_plates = {}


# ============================================================
# PROCESS VIDEO
# ============================================================

print()
print("=" * 60)
print("STARTING YOLO11 + OCR + DATABASE PROCESSING")
print("=" * 60)


frame_number = 0


while True:

    success, frame = cap.read()

    if not success:
        break

    frame_number += 1


    # ========================================================
    # YOLO DETECTION
    # ========================================================

    results = model.predict(
        source=frame,
        conf=YOLO_CONFIDENCE,
        imgsz=640,
        verbose=False
    )


    result = results[0]


    # ========================================================
    # PROCESS EACH LICENSE PLATE
    # ========================================================

    if result.boxes is not None:

        for box in result.boxes:

            yolo_conf = float(
                box.conf[0]
            )


            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )


            # Keep coordinates inside image
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)


            if x2 <= x1 or y2 <= y1:
                continue


            # =================================================
            # CROP LICENSE PLATE
            # =================================================

            plate_crop = frame[
                y1:y2,
                x1:x2
            ]


            if plate_crop.size == 0:
                continue


            # =================================================
            # OCR
            # =================================================

            try:

                ocr_results = reader.readtext(
                    plate_crop
                )

            except Exception:

                continue


            if not ocr_results:
                continue


            # =================================================
            # SELECT BEST OCR RESULT
            # =================================================

            best_text = ""

            best_ocr_conf = 0.0


            for detection in ocr_results:

                text = detection[1]

                ocr_conf = float(
                    detection[2]
                )


                normalized = normalize_plate(
                    text
                )


                if (
                    ocr_conf > best_ocr_conf
                    and valid_plate(normalized)
                ):

                    best_text = normalized

                    best_ocr_conf = ocr_conf


            if not best_text:
                continue


            # =================================================
            # OCR CONFIDENCE CHECK
            # =================================================

            if best_ocr_conf < OCR_CONFIDENCE:
                continue


            # =================================================
            # STORE OBSERVATION
            # =================================================

            if best_text not in observations:

                observations[best_text] = []


            observations[best_text].append(
                best_ocr_conf
            )


            count = len(
                observations[best_text]
            )


            # =================================================
            # PRINT OBSERVATION
            # =================================================

            print()
            print("-" * 60)

            print(
                f"Frame       : {frame_number}"
            )

            print(
                f"OCR Reading : {best_text}"
            )

            print(
                f"OCR Conf.   : {best_ocr_conf:.2f}"
            )

            print(
                f"YOLO Conf.  : {yolo_conf:.2f}"
            )

            print(
                f"Observations: {count}"
            )


            # =================================================
            # CONFIRM PLATE
            # =================================================

            if count >= MIN_OBSERVATIONS:


                # Calculate average OCR confidence

                average_confidence = (
                    sum(observations[best_text])
                    /
                    len(observations[best_text])
                )


                # =================================================
                # CHECK SAVE COOLDOWN
                # =================================================

                previous_frame = (
                    last_saved_frame.get(
                        best_text,
                        -999999
                    )
                )


                if (
                    frame_number
                    -
                    previous_frame
                    >=
                    SAVE_COOLDOWN_FRAMES
                ):


                    # =================================================
                    # TIMESTAMP
                    # =================================================

                    seconds = (
                        frame_number / fps
                        if fps > 0
                        else 0
                    )


                    timestamp = (
                        f"{int(seconds // 3600)}:"
                        f"{int((seconds % 3600) // 60):02d}:"
                        f"{int(seconds % 60):02d}"
                    )


                    # =================================================
                    # SAVE DATABASE RECORD
                    # =================================================

                    save_to_database(

                        plate_number=best_text,

                        camera_id=CAMERA_ID,

                        city=CITY,

                        road=ROAD,

                        timestamp=timestamp,

                        frame=frame_number,

                        confidence=average_confidence,

                        video_source=os.path.basename(
                            VIDEO_PATH
                        )

                    )


                    last_saved_frame[
                        best_text
                    ] = frame_number


                    confirmed_plates[
                        best_text
                    ] = {

                        "frame": frame_number,

                        "timestamp": timestamp,

                        "confidence":
                            average_confidence

                    }


                    # =================================================
                    # CONFIRMATION MESSAGE
                    # =================================================

                    print()
                    print("=" * 60)
                    print("LICENSE PLATE CONFIRMED + SAVED")
                    print("=" * 60)

                    print(
                        f"Plate       : {best_text}"
                    )

                    print(
                        f"OCR Avg.    : "
                        f"{average_confidence:.2f}"
                    )

                    print(
                        f"YOLO Conf.  : "
                        f"{yolo_conf:.2f}"
                    )

                    print(
                        f"Camera      : {CAMERA_ID}"
                    )

                    print(
                        f"City        : {CITY}"
                    )

                    print(
                        f"Road        : {ROAD}"
                    )

                    print(
                        f"Timestamp   : {timestamp}"
                    )

                    print(
                        f"Frame       : {frame_number}"
                    )

                    print(
                        "Database    : Record saved successfully."
                    )

                    print("=" * 60)


            # =================================================
            # DRAW BOX
            # =================================================

            cv2.rectangle(

                frame,

                (x1, y1),

                (x2, y2),

                (0, 255, 0),

                2

            )


            # =================================================
            # DRAW OCR TEXT
            # =================================================

            label = (
                f"{best_text} "
                f"{best_ocr_conf:.2f}"
            )


            cv2.putText(

                frame,

                label,

                (x1, max(30, y1 - 10)),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (0, 255, 0),

                2

            )


    # ========================================================
    # WRITE OUTPUT FRAME
    # ========================================================

    writer.write(frame)


    # ========================================================
    # PROGRESS
    # ========================================================

    if (
        frame_number % 20 == 0
        or frame_number == total_frames
    ):

        percentage = (
            frame_number
            /
            total_frames
            *
            100
        )

        print(
            f"Processing: "
            f"{percentage:.1f}% "
            f"({frame_number}/{total_frames})",
            end="\r"
        )


# ============================================================
# RELEASE VIDEO
# ============================================================

cap.release()

writer.release()


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print()
print("=" * 60)
print("VIDEO PROCESSING COMPLETED")
print("=" * 60)

print(
    f"Confirmed plates: "
    f"{len(confirmed_plates)}"
)


if confirmed_plates:

    print()
    print("=" * 60)
    print("CONFIRMED PLATE SUMMARY")
    print("=" * 60)


    for plate, information in confirmed_plates.items():

        print()

        print(
            f"Plate       : {plate}"
        )

        print(
            f"Frame       : "
            f"{information['frame']}"
        )

        print(
            f"Timestamp   : "
            f"{information['timestamp']}"
        )

        print(
            f"Confidence  : "
            f"{information['confidence']:.2f}"
        )


else:

    print()
    print(
        "No plate met the confirmation threshold."
    )


# ============================================================
# OUTPUT FILES
# ============================================================

print()
print("=" * 60)
print("OUTPUT FILES")
print("=" * 60)

print(
    f"Output Video:"
)

print(
    OUTPUT_VIDEO
)

print()

print(
    "Database:"
)

print(
    DATABASE_PATH
)

print("=" * 60)

print()
print("Processing finished successfully.")