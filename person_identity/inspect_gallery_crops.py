import os
import cv2

from ultralytics import YOLO


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

PERSON_IDENTITY_DIR = os.path.join(
    BASE_DIR,
    "person_identity"
)

PERSONS_DIR = os.path.join(
    PERSON_IDENTITY_DIR,
    "persons"
)

CROPS_DIR = os.path.join(
    PERSON_IDENTITY_DIR,
    "debug_crops"
)

YOLO_CONFIDENCE = 0.45
YOLO_IMAGE_SIZE = 640

BBOX_EXPANSION = 0.10

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    CROPS_DIR,
    exist_ok=True
)


# ============================================================
# LOAD YOLO
# ============================================================

print("=" * 70)
print("PERSON REID CROP INSPECTION")
print("=" * 70)

print("")
print("Loading YOLO...")

model = YOLO(
    "yolo11n.pt"
)

print("YOLO loaded successfully.")


# ============================================================
# EXPAND BOUNDING BOX
# ============================================================

def expand_bbox(
    coordinates,
    image_shape,
    expansion=0.10
):

    x1, y1, x2, y2 = map(
        int,
        coordinates
    )

    height, width = image_shape[:2]

    box_width = x2 - x1
    box_height = y2 - y1

    expand_x = int(
        box_width * expansion
    )

    expand_y = int(
        box_height * expansion
    )

    x1 -= expand_x
    y1 -= expand_y

    x2 += expand_x
    y2 += expand_y

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

    return (
        x1,
        y1,
        x2,
        y2
    )


# ============================================================
# PROCESS IMAGE
# ============================================================

def process_image(
    person_id,
    image_name,
    image_path
):

    print("")
    print(
        f"Processing: "
        f"{person_id}/{image_name}"
    )

    image = cv2.imread(
        image_path
    )

    if image is None:

        print(
            "  ERROR: Could not read image."
        )

        return

    results = model(
        image,
        classes=[0],
        conf=YOLO_CONFIDENCE,
        imgsz=YOLO_IMAGE_SIZE,
        verbose=False
    )

    persons = []

    for result in results:

        if result.boxes is None:
            continue

        for box in result.boxes:

            confidence = float(
                box.conf[0]
            )

            coordinates = (
                box.xyxy[0]
                .cpu()
                .numpy()
            )

            persons.append(
                (
                    coordinates,
                    confidence
                )
            )

    if not persons:

        print(
            "  NO PERSON DETECTED"
        )

        return

    best_box, best_confidence = max(
        persons,
        key=lambda x: x[1]
    )

    x1, y1, x2, y2 = expand_bbox(
        best_box,
        image.shape,
        BBOX_EXPANSION
    )

    crop = image[
        y1:y2,
        x1:x2
    ]

    if crop.size == 0:

        print(
            "  ERROR: Empty crop."
        )

        return

    crop_height, crop_width = (
        crop.shape[:2]
    )

    aspect_ratio = (
        crop_height / crop_width
        if crop_width > 0
        else 0
    )

    print(
        f"  YOLO confidence : "
        f"{best_confidence:.3f}"
    )

    print(
        f"  Crop size       : "
        f"{crop_width} x {crop_height}"
    )

    print(
        f"  Aspect ratio    : "
        f"{aspect_ratio:.3f}"
    )

    output_name = (
        f"{person_id}__{image_name}"
    )

    output_path = os.path.join(
        CROPS_DIR,
        output_name
    )

    cv2.imwrite(
        output_path,
        crop
    )

    print(
        f"  Saved crop      : "
        f"{output_path}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    person_folders = []

    for item in os.listdir(
        PERSONS_DIR
    ):

        path = os.path.join(
            PERSONS_DIR,
            item
        )

        if os.path.isdir(
            path
        ):

            person_folders.append(
                item
            )

    person_folders.sort()

    print("")
    print(
        f"People found: "
        f"{len(person_folders)}"
    )

    total = 0

    for person_id in person_folders:

        person_directory = os.path.join(
            PERSONS_DIR,
            person_id
        )

        for filename in sorted(
            os.listdir(
                person_directory
            )
        ):


            if not filename.lower().endswith(
                IMAGE_EXTENSIONS
            ):

                continue

            total += 1

            image_path = os.path.join(
                person_directory,
                filename
            )

            process_image(
                person_id,
                filename,
                image_path
            )

    print("")
    print("=" * 70)
    print("CROP INSPECTION COMPLETED")
    print("=" * 70)

    print("")
    print(
        f"Total images: {total}"
    )

    print("")
    print(
        "Open this folder:"
    )

    print(
        CROPS_DIR
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()