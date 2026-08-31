from ultralytics import YOLO
import easyocr
import cv2
import os
import re

# ==========================================
# Paths
# ==========================================

MODEL_PATH = "plate_detector/models/best.pt"
IMAGE_PATH = "plate_detector/test_images/test_car.jpg"

OUTPUT_DIR = "plate_detector/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==========================================
# Load YOLO11 License Plate Detector
# ==========================================

print("=" * 60)
print("Loading License Plate Detector...")
print("=" * 60)

model = YOLO(MODEL_PATH)

# ==========================================
# Load EasyOCR
# ==========================================

print("Loading OCR model...")

reader = easyocr.Reader(
    ['en'],
    gpu=False
)

print("OCR loaded successfully!")

# ==========================================
# Read Image
# ==========================================

image = cv2.imread(IMAGE_PATH)

if image is None:
    print("ERROR: Image could not be loaded.")
    exit()

# ==========================================
# Detect License Plate
# ==========================================

results = model.predict(
    source=image,
    conf=0.40,
    imgsz=640,
    verbose=False
)

result = results[0]

# ==========================================
# Check Detection
# ==========================================

if result.boxes is None or len(result.boxes) == 0:

    print("=" * 60)
    print("No license plate detected.")
    print("=" * 60)
    exit()

# ==========================================
# Process Detected Plates
# ==========================================

for i, box in enumerate(result.boxes):

    confidence = float(box.conf[0])

    x1, y1, x2, y2 = map(
        int,
        box.xyxy[0]
    )

    print("\n" + "=" * 60)
    print(f"LICENSE PLATE #{i + 1}")
    print("=" * 60)

    print(f"Detection Confidence: {confidence:.2f}")
    print(f"Coordinates: ({x1}, {y1}, {x2}, {y2})")

    # --------------------------------------
    # Crop License Plate
    # --------------------------------------

    plate_crop = image[y1:y2, x1:x2]

    if plate_crop.size == 0:
        print("Invalid plate crop.")
        continue

    # --------------------------------------
    # Save Plate Crop
    # --------------------------------------

    crop_path = os.path.join(
        OUTPUT_DIR,
        f"plate_{i + 1}.jpg"
    )

    cv2.imwrite(
        crop_path,
        plate_crop
    )

    print(f"Plate crop saved: {crop_path}")

    # --------------------------------------
    # Resize Plate
    # --------------------------------------

    height, width = plate_crop.shape[:2]

    scale = 4

    enlarged_plate = cv2.resize(
        plate_crop,
        (width * scale, height * scale),
        interpolation=cv2.INTER_CUBIC
    )

    # --------------------------------------
    # OCR
    # --------------------------------------

    ocr_results = reader.readtext(
        enlarged_plate
    )

    print("\nOCR Results:")

    if not ocr_results:

        print("No text detected.")

    else:

        detected_texts = []

        for detection in ocr_results:

            bbox, text, ocr_confidence = detection

            text = text.strip()

            print(
                f"Text: {text} | "
                f"Confidence: {ocr_confidence:.2f}"
            )

            detected_texts.append(text)

        # ----------------------------------
        # Combine detected text
        # ----------------------------------

        plate_number = " ".join(
            detected_texts
        )

        # Remove unwanted characters
        plate_number = re.sub(
            r'[^A-Za-z0-9-]',
            '',
            plate_number
        )

        print("\n" + "=" * 60)
        print("FINAL PLATE NUMBER")
        print("=" * 60)

        print(plate_number)

# ==========================================
# Finished
# ==========================================

print("\n" + "=" * 60)
print("PLATE DETECTION + OCR COMPLETED")
print("=" * 60)