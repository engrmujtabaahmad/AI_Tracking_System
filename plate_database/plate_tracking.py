import os
import re
import csv
from collections import defaultdict
from datetime import timedelta

import cv2
import easyocr
from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_PATH = "assets/Traffic1.mp4"

MODEL_PATH = "plate_detector/models/best.pt"

OUTPUT_DIR = "plate_detector/outputs"
OUTPUT_VIDEO = os.path.join(
    OUTPUT_DIR,
    "Traffic1_tracked_plates.mp4"
)

DATABASE_PATH = "plate_database/plate_records.csv"

CAMERA_ID = "Camera_01"
CITY = "Lahore"
ROAD = "Canal Road"

YOLO_CONFIDENCE = 0.40
OCR_CONFIDENCE = 0.30

# Process every Nth frame
FRAME_SKIP = 2

# Minimum number of observations required
MIN_OBSERVATIONS = 2


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("plate_database", exist_ok=True)


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
# NORMALIZE OCR TEXT
# ============================================================

def normalize_plate(text):
    """
    Clean OCR output so that minor OCR differences
    don't create completely different plate numbers.
    """

    text = text.upper()

    # Keep only letters and numbers
    text = re.sub(r"[^A-Z0-9]", "", text)

    # Remove extremely short readings
    if len(text) < 3:
        return ""

    return text


# ============================================================
# SIMILARITY CHECK
# ============================================================

def plate_similarity(a, b):
    """
    Simple character-based similarity.
    """

    if not a or not b:
        return 0.0

    # Exact match
    if a == b:
        return 1.0

    # Compare common length
    min_len = min(len(a), len(b))

    matches = 0

    for i in range(min_len):
        if a[i] == b[i]:
            matches += 1

    similarity = matches / max(len(a), len(b))

    return similarity


# ============================================================
# FIND EXISTING PLATE GROUP
# ============================================================

def find_matching_plate(text, plate_groups):
    """
    Check whether OCR text belongs to an existing
    plate group.
    """

    best_match = None
    best_score = 0

    for plate in plate_groups:

        score = plate_similarity(
            text,
            plate
        )

        if score > best_score:
            best_score = score
            best_match = plate

    if best_score >= 0.60:
        return best_match

    return None


# ============================================================
# VIDEO OPEN
# ============================================================

print()
print("=" * 60)
print("OPENING VIDEO")
print("=" * 60)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():

    print("ERROR: Could not open video.")

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

duration = total_frames / fps if fps else 0


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
# OUTPUT VIDEO
# ============================================================

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    fourcc,
    fps,
    (width, height)
)


# ============================================================
# PLATE OBSERVATIONS
# ============================================================

plate_groups = defaultdict(list)

observation_data = []

frame_number = 0


# ============================================================
# PROCESS VIDEO
# ============================================================

print()
print("=" * 60)
print("STARTING MULTI-FRAME PLATE TRACKING")
print("=" * 60)


while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    # --------------------------------------------------------
    # Run detection on selected frames
    # --------------------------------------------------------

    if frame_number % FRAME_SKIP != 0:

        out.write(frame)

        continue


    # --------------------------------------------------------
    # YOLO detection
    # --------------------------------------------------------

    results = model.predict(
        source=frame,
        conf=YOLO_CONFIDENCE,
        imgsz=640,
        verbose=False
    )

    result = results[0]


    # --------------------------------------------------------
    # Draw YOLO results
    # --------------------------------------------------------

    annotated = frame.copy()


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


            # ------------------------------------------------
            # Crop plate
            # ------------------------------------------------

            plate_crop = frame[
                y1:y2,
                x1:x2
            ]


            if plate_crop.size == 0:
                continue


            # ------------------------------------------------
            # OCR
            # ------------------------------------------------

            ocr_results = reader.readtext(
                plate_crop
            )


            if not ocr_results:
                continue


            best_text = ""
            best_ocr_conf = 0


            for _, text, confidence in ocr_results:

                cleaned = normalize_plate(
                    text
                )

                if not cleaned:
                    continue

                if confidence > best_ocr_conf:

                    best_text = cleaned
                    best_ocr_conf = confidence


            # ------------------------------------------------
            # Ignore weak OCR
            # ------------------------------------------------

            if not best_text:
                continue

            if best_ocr_conf < OCR_CONFIDENCE:
                continue


            # ------------------------------------------------
            # Find matching plate group
            # ------------------------------------------------

            matched_plate = find_matching_plate(
                best_text,
                plate_groups
            )


            if matched_plate is None:

                matched_plate = best_text


            # ------------------------------------------------
            # Store observation
            # ------------------------------------------------

            plate_groups[
                matched_plate
            ].append({

                "frame": frame_number,

                "ocr": best_text,

                "ocr_confidence": best_ocr_conf,

                "yolo_confidence": yolo_conf

            })


            observation_data.append({

                "plate_group": matched_plate,

                "ocr": best_text,

                "ocr_confidence": best_ocr_conf,

                "yolo_confidence": yolo_conf,

                "frame": frame_number

            })


            # ------------------------------------------------
            # Draw result
            # ------------------------------------------------

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


            label = (
                f"{best_text} "
                f"{best_ocr_conf:.2f}"
            )


            cv2.putText(
                annotated,
                label,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )


            print()
            print("-" * 60)

            print(
                f"Frame       : {frame_number}"
            )

            print(
                f"OCR Reading : {best_text}"
            )

            print(
                f"OCR Conf.   : "
                f"{best_ocr_conf:.2f}"
            )

            print(
                f"YOLO Conf.  : "
                f"{yolo_conf:.2f}"
            )


    # --------------------------------------------------------
    # Write frame
    # --------------------------------------------------------

    out.write(annotated)


    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if frame_number % 20 == 0:

        progress = (
            frame_number /
            total_frames
        ) * 100

        print(
            f"Processing: "
            f"{progress:.1f}% "
            f"({frame_number}/{total_frames})"
        )


# ============================================================
# RELEASE VIDEO
# ============================================================

cap.release()
out.release()


# ============================================================
# SELECT CONFIRMED PLATES
# ============================================================

print()
print("=" * 60)
print("PROCESSING COMPLETED")
print("=" * 60)

print(
    f"Total OCR observations: "
    f"{len(observation_data)}"
)


confirmed_plates = []


for plate, observations in plate_groups.items():

    count = len(observations)

    average_confidence = sum(
        x["ocr_confidence"]
        for x in observations
    ) / count


    # --------------------------------------------------------
    # Confirmation rule
    # --------------------------------------------------------

    if (
        count >= MIN_OBSERVATIONS
        and average_confidence >= OCR_CONFIDENCE
    ):

        confirmed_plates.append({

            "plate": plate,

            "count": count,

            "average_confidence":
                average_confidence,

            "observations":
                observations

        })


# ============================================================
# DISPLAY SUMMARY
# ============================================================

print()
print("=" * 60)
print("PLATE GROUP SUMMARY")
print("=" * 60)


for plate, observations in plate_groups.items():

    avg = sum(
        x["ocr_confidence"]
        for x in observations
    ) / len(observations)


    print(
        f"{plate:<15}"
        f" Count: {len(observations):<3}"
        f" Avg Confidence: {avg:.2f}"
    )


# ============================================================
# SAVE CONFIRMED PLATES
# ============================================================

if not confirmed_plates:

    print()
    print("No plate was confirmed.")

else:

    print()
    print("=" * 60)
    print("CONFIRMED PLATES")
    print("=" * 60)


    # --------------------------------------------------------
    # Create database if necessary
    # --------------------------------------------------------

    file_exists = os.path.exists(
        DATABASE_PATH
    )


    with open(
        DATABASE_PATH,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)


        # ----------------------------------------------------
        # Header
        # ----------------------------------------------------

        if not file_exists:

            writer.writerow([

                "plate_number",
                "camera_id",
                "city",
                "road",
                "first_seen",
                "last_seen",
                "first_frame",
                "last_frame",
                "observations",
                "average_confidence"

            ])


        # ----------------------------------------------------
        # Save each confirmed plate
        # ----------------------------------------------------

        for item in confirmed_plates:

            plate = item["plate"]

            observations = item[
                "observations"
            ]


            # ------------------------------------------------
            # Best OCR reading
            # ------------------------------------------------

            best_observation = max(
                observations,
                key=lambda x:
                    x["ocr_confidence"]
            )


            # ------------------------------------------------
            # Frame information
            # ------------------------------------------------

            first_frame = min(
                x["frame"]
                for x in observations
            )

            last_frame = max(
                x["frame"]
                for x in observations
            )


            # ------------------------------------------------
            # Convert frames to time
            # ------------------------------------------------

            first_seconds = (
                first_frame / fps
            )

            last_seconds = (
                last_frame / fps
            )


            first_time = str(
                timedelta(
                    seconds=int(
                        first_seconds
                    )
                )
            )


            last_time = str(
                timedelta(
                    seconds=int(
                        last_seconds
                    )
                )
            )


            # ------------------------------------------------
            # Write database record
            # ------------------------------------------------

            writer.writerow([

                plate,

                CAMERA_ID,

                CITY,

                ROAD,

                first_time,

                last_time,

                first_frame,

                last_frame,

                len(observations),

                round(
                    item[
                        "average_confidence"
                    ],
                    2
                )

            ])


            print()
            print(
                f"✓ {plate}"
            )

            print(
                f"  Observations : "
                f"{len(observations)}"
            )

            print(
                f"  First seen   : "
                f"{first_time}"
            )

            print(
                f"  Last seen    : "
                f"{last_time}"
            )

            print(
                f"  Last frame   : "
                f"{last_frame}"
            )

            print(
                f"  Confidence   : "
                f"{item['average_confidence']:.2f}"
            )


# ============================================================
# FINAL OUTPUT
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

print()
print("=" * 60)
print("PLATE TRACKING FINISHED SUCCESSFULLY")
print("=" * 60)