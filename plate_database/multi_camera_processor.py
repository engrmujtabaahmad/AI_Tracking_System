import os
import cv2
import sqlite3
import re
from collections import defaultdict
from ultralytics import YOLO
import easyocr


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "plate_detector",
    "models",
    "best.pt"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "plate_detector",
    "outputs"
)

DATABASE_PATH = os.path.join(
    BASE_DIR,
    "plate_database",
    "traffic_tracking.db"
)

VIDEO_NAME = "Traffic1.mp4"

CAMERA_ID = "Camera_01"
CITY = "Lahore"
ROAD = "Canal Road"

# Detection settings
YOLO_CONFIDENCE = 0.45
OCR_CONFIDENCE = 0.10

# Multi-frame confirmation settings
MIN_OBSERVATIONS = 3
MIN_AVG_OCR = 0.28
MIN_BEST_OCR = 0.38

# Process every frame
FRAME_SKIP = 1


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# HELPER: FIND VIDEO
# ============================================================

def find_video(video_name):

    print("\nSearching for video...")

    for root, dirs, files in os.walk(BASE_DIR):

        if video_name in files:

            video_path = os.path.join(
                root,
                video_name
            )

            print("Video found:")
            print(video_path)

            return video_path

    return None


# ============================================================
# HELPER: CLEAN OCR TEXT
# ============================================================

def clean_plate_text(text):

    if text is None:
        return ""

    text = str(text).upper()

    # Remove spaces, hyphens and special characters
    text = re.sub(
        r"[^A-Z0-9]",
        "",
        text
    )

    return text


# ============================================================
# HELPER: NORMALIZE OCR VARIATIONS
# ============================================================

def normalize_plate(text):

    text = clean_plate_text(text)

    if not text:
        return ""

    # Common OCR character corrections
    replacements = {
        "O": "0",
        "I": "1",
        "L": "1"
    }

    # We don't blindly replace everything because
    # Pakistani plates can contain letters.
    #
    # Instead, keep the original cleaned text.
    return text


# ============================================================
# HELPER: CHECK WHETHER TWO PLATES ARE SIMILAR
# ============================================================

def plate_similarity(text1, text2):

    text1 = normalize_plate(text1)
    text2 = normalize_plate(text2)

    if not text1 or not text2:
        return False

    # Exact match
    if text1 == text2:
        return True

    # One text contained inside the other
    if text1 in text2 or text2 in text1:

        difference = abs(
            len(text1) - len(text2)
        )

        if difference <= 2:
            return True

    # Character comparison
    max_length = max(
        len(text1),
        len(text2)
    )

    if max_length == 0:
        return False

    same_positions = 0

    for a, b in zip(text1, text2):

        if a == b:
            same_positions += 1

    similarity = same_positions / max_length

    return similarity >= 0.60


# ============================================================
# HELPER: GROUP OCR OBSERVATIONS
# ============================================================

def find_matching_group(
    groups,
    plate_text
):

    for group_name in groups:

        if plate_similarity(
            group_name,
            plate_text
        ):

            return group_name

    return None


# ============================================================
# DATABASE SETUP
# ============================================================

def setup_database():

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS vehicle_detections (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            plate_number TEXT NOT NULL,

            camera_id TEXT,

            city TEXT,

            road TEXT,

            timestamp TEXT,

            frame INTEGER,

            confidence REAL,

            video_source TEXT

        )
    """)

    connection.commit()

    connection.close()


# ============================================================
# SAVE DATABASE RECORD
# ============================================================

def save_detection(
    plate_number,
    timestamp,
    frame_number,
    confidence,
    video_source
):

    connection = sqlite3.connect(
        DATABASE_PATH
    )

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
        CAMERA_ID,
        CITY,
        ROAD,
        timestamp,
        frame_number,
        confidence,
        video_source

    ))

    connection.commit()

    connection.close()


# ============================================================
# MAIN PROGRAM
# ============================================================

print("=" * 60)
print("SMART TRAFFIC MONITORING SYSTEM")
print("MULTI-CAMERA MULTI-FRAME PROCESSOR")
print("=" * 60)


# ============================================================
# LOAD YOLO
# ============================================================

print("\nLoading YOLO11 license plate detector...")

try:

    model = YOLO(
        MODEL_PATH
    )

    print("YOLO11 model loaded successfully.")

except Exception as e:

    print("\nERROR loading YOLO model:")
    print(e)
    exit()


# ============================================================
# LOAD OCR
# ============================================================

print("\nLoading EasyOCR...")

try:

    reader = easyocr.Reader(
        ["en"],
        gpu=False
    )

    print("OCR loaded successfully.")

except Exception as e:

    print("\nERROR loading OCR:")
    print(e)
    exit()


# ============================================================
# DATABASE
# ============================================================

print("\nConnecting to tracking database...")

setup_database()

print("Database ready.")


# ============================================================
# FIND VIDEO
# ============================================================

video_path = find_video(
    VIDEO_NAME
)

if video_path is None:

    print("\nERROR: Video not found.")
    print(
        f"Please make sure {VIDEO_NAME} "
        "exists somewhere inside the project."
    )

    exit()


# ============================================================
# AVAILABLE CAMERAS
# ============================================================

print("\n")
print("=" * 60)
print("AVAILABLE CAMERA VIDEOS")
print("=" * 60)

print(
    f"{CAMERA_ID} -> {VIDEO_NAME}"
)


# ============================================================
# PROCESS CAMERA
# ============================================================

print("\n")
print("=" * 60)
print("PROCESSING CAMERA")
print("=" * 60)

print(
    f"Camera : {CAMERA_ID}"
)

print(
    f"Video  : {VIDEO_NAME}"
)

print(
    f"City   : {CITY}"
)

print(
    f"Road   : {ROAD}"
)


# ============================================================
# OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(
    video_path
)

if not cap.isOpened():

    print("\nERROR: Could not open video.")

    exit()


# ============================================================
# VIDEO INFORMATION
# ============================================================

fps = cap.get(
    cv2.CAP_PROP_FPS
)

width = int(
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)

height = int(
    cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)

total_frames = int(
    cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)

duration = (
    total_frames / fps
    if fps > 0
    else 0
)


print("\nVIDEO INFORMATION")
print("-" * 60)

print(
    f"FPS        : {fps:.2f}"
)

print(
    f"Resolution : {width} x {height}"
)

print(
    f"Frames     : {total_frames}"
)

print(
    f"Duration   : {duration:.2f} seconds"
)


# ============================================================
# OUTPUT VIDEO
# ============================================================

output_path = os.path.join(
    OUTPUT_DIR,
    f"{CAMERA_ID}_{os.path.splitext(VIDEO_NAME)[0]}_processed.mp4"
)


fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

writer = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (width, height)
)


# ============================================================
# OCR GROUPS
# ============================================================

plate_groups = defaultdict(
    list
)

total_observations = 0


# ============================================================
# PROCESS VIDEO
# ============================================================

print("\n")
print("=" * 60)
print("STARTING MULTI-FRAME PROCESSING")
print("=" * 60)


frame_number = 0


while True:

    success, frame = cap.read()

    if not success:
        break

    frame_number += 1


    # --------------------------------------------------------
    # FRAME SKIP
    # --------------------------------------------------------

    if (
        frame_number % FRAME_SKIP
        != 0
    ):

        writer.write(frame)

        continue


    # --------------------------------------------------------
    # YOLO DETECTION
    # --------------------------------------------------------

    results = model.predict(
        source=frame,
        conf=YOLO_CONFIDENCE,
        imgsz=640,
        verbose=False
    )


    result = results[0]


    # --------------------------------------------------------
    # PROCESS DETECTED PLATES
    # --------------------------------------------------------

    if (
        result.boxes is not None
        and len(result.boxes) > 0
    ):

        for box in result.boxes:


            # ------------------------------------------------
            # YOLO CONFIDENCE
            # ------------------------------------------------

            yolo_conf = float(
                box.conf[0]
            )


            # ------------------------------------------------
            # PLATE COORDINATES
            # ------------------------------------------------

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )


            # Keep coordinates inside image
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


            # ------------------------------------------------
            # CROP PLATE
            # ------------------------------------------------

            plate_crop = frame[
                y1:y2,
                x1:x2
            ]


            if plate_crop.size == 0:
                continue


            # ------------------------------------------------
            # UPSCALE PLATE
            # ------------------------------------------------

            plate_crop = cv2.resize(
                plate_crop,
                None,
                fx=3,
                fy=3,
                interpolation=cv2.INTER_CUBIC
            )


            # ------------------------------------------------
            # OCR
            # ------------------------------------------------

            try:

                ocr_results = reader.readtext(
                    plate_crop
                )

            except Exception:
                continue


            if not ocr_results:
                continue


            # ------------------------------------------------
            # GET BEST OCR RESULT
            # ------------------------------------------------

            best_text = ""
            best_conf = 0.0


            for detection in ocr_results:

                if len(detection) < 3:
                    continue


                text = detection[1]
                confidence = float(
                    detection[2]
                )


                text = clean_plate_text(
                    text
                )


                if not text:
                    continue


                if confidence > best_conf:

                    best_text = text
                    best_conf = confidence


            if not best_text:
                continue


            if best_conf < OCR_CONFIDENCE:
                continue


            # ------------------------------------------------
            # FIND EXISTING OCR GROUP
            # ------------------------------------------------

            matching_group = find_matching_group(
                plate_groups,
                best_text
            )


            if matching_group is None:

                group_name = best_text

                plate_groups[
                    group_name
                ].append({

                    "text": best_text,

                    "ocr_conf": best_conf,

                    "yolo_conf": yolo_conf,

                    "frame": frame_number,

                    "x1": x1,

                    "y1": y1,

                    "x2": x2,

                    "y2": y2

                })

            else:

                plate_groups[
                    matching_group
                ].append({

                    "text": best_text,

                    "ocr_conf": best_conf,

                    "yolo_conf": yolo_conf,

                    "frame": frame_number,

                    "x1": x1,

                    "y1": y1,

                    "x2": x2,

                    "y2": y2

                })


            total_observations += 1


            # ------------------------------------------------
            # PRINT OBSERVATION
            # ------------------------------------------------

            print(
                f"\nCamera: {CAMERA_ID}"
            )

            print(
                f"Frame: {frame_number}"
            )

            print(
                f"OCR: {best_text}"
            )

            print(
                f"OCR Confidence: {best_conf:.2f}"
            )

            print(
                f"YOLO Confidence: {yolo_conf:.2f}"
            )


            # ------------------------------------------------
            # DRAW BOX
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


            cv2.putText(
                frame,
                f"{best_text} {best_conf:.2f}",
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )


    # --------------------------------------------------------
    # PROGRESS
    # --------------------------------------------------------

    if (
        frame_number % 20 == 0
        or frame_number == total_frames
    ):

        progress = (
            frame_number
            / total_frames
            * 100
        )

        print(
            f"Processing: "
            f"{progress:.1f}% "
            f"({frame_number}/{total_frames})"
        )


    # --------------------------------------------------------
    # WRITE FRAME
    # --------------------------------------------------------

    writer.write(
        frame
    )


# ============================================================
# RELEASE VIDEO
# ============================================================

cap.release()
writer.release()


# ============================================================
# PROCESS OCR GROUPS
# ============================================================

print("\n")
print("=" * 60)
print("VIDEO PROCESSING COMPLETED")
print("=" * 60)

print(
    f"Total OCR observations: "
    f"{total_observations}"
)


print("\n")
print("=" * 60)
print("OCR GROUP SUMMARY")
print("=" * 60)


confirmed_plates = []


for group_name, observations in plate_groups.items():


    count = len(
        observations
    )


    avg_ocr = sum(
        obs["ocr_conf"]
        for obs in observations
    ) / count


    avg_yolo = sum(
        obs["yolo_conf"]
        for obs in observations
    ) / count


    best_observation = max(
        observations,
        key=lambda x: x["ocr_conf"]
    )


    best_ocr = (
        best_observation["ocr_conf"]
    )


    frames = [
        obs["frame"]
        for obs in observations
    ]


    print(
        f"{group_name:<15}"
        f"Count: {count:<3}"
        f"Avg OCR: {avg_ocr:.2f}"
        f"  Best OCR: {best_ocr:.2f}"
    )


    # ========================================================
    # SMART CONFIRMATION
    # ========================================================

    if (
        count >= MIN_OBSERVATIONS
        and (
            avg_ocr >= MIN_AVG_OCR
            or best_ocr >= MIN_BEST_OCR
        )
    ):

        # Use the best OCR reading
        # as the final plate number.

        final_plate = clean_plate_text(
            best_observation["text"]
        )


        # Average YOLO confidence
        final_confidence = avg_yolo


        # First and last frames
        first_frame = min(
            frames
        )

        last_frame = max(
            frames
        )


        # Timestamp
        timestamp_seconds = (
            last_frame / fps
            if fps > 0
            else 0
        )


        timestamp = (
            str(
                __import__("datetime").timedelta(
                    seconds=int(
                        timestamp_seconds
                    )
                )
            )
        )


        confirmed_data = {

            "plate": final_plate,

            "count": count,

            "avg_ocr": avg_ocr,

            "best_ocr": best_ocr,

            "avg_yolo": avg_yolo,

            "first_frame": first_frame,

            "last_frame": last_frame,

            "timestamp": timestamp

        }


        confirmed_plates.append(
            confirmed_data
        )


# ============================================================
# CONFIRMED PLATES
# ============================================================

print("\n")
print("=" * 60)
print("CONFIRMED PLATES")
print("=" * 60)


if not confirmed_plates:

    print(
        "\nNo plates confirmed."
    )

else:

    for plate in confirmed_plates:

        print(
            f"\n✓ {plate['plate']}"
        )

        print(
            f"  Observations : "
            f"{plate['count']}"
        )

        print(
            f"  Average OCR  : "
            f"{plate['avg_ocr']:.2f}"
        )

        print(
            f"  Best OCR     : "
            f"{plate['best_ocr']:.2f}"
        )

        print(
            f"  Average YOLO : "
            f"{plate['avg_yolo']:.2f}"
        )

        print(
            f"  First Frame  : "
            f"{plate['first_frame']}"
        )

        print(
            f"  Last Frame   : "
            f"{plate['last_frame']}"
        )

        print(
            f"  Timestamp    : "
            f"{plate['timestamp']}"
        )


        # ====================================================
        # SAVE TO DATABASE
        # ====================================================

        save_detection(

            plate_number=plate["plate"],

            timestamp=plate["timestamp"],

            frame_number=plate["last_frame"],

            confidence=plate["avg_ocr"],

            video_source=VIDEO_NAME

        )

        print(
            "  Database     : "
            "Record saved successfully."
        )


# ============================================================
# OUTPUT INFORMATION
# ============================================================

print("\n")
print("=" * 60)
print("OUTPUT FILES")
print("=" * 60)

print(
    "Output video:"
)

print(
    output_path
)

print(
    "\nDatabase:"
)

print(
    DATABASE_PATH
)

print("=" * 60)


print("\n")
print("=" * 60)
print("MULTI-CAMERA PROCESSING FINISHED")
print("=" * 60)

print(
    f"Database: {DATABASE_PATH}"
)

print(
    f"Outputs : {OUTPUT_DIR}"
)

print("=" * 60)