import os
import cv2
import sqlite3
import numpy as np
import torch
import torchreid

from ultralytics import YOLO
from PIL import Image


# ============================================================
# PERSON IDENTITY GALLERY BUILDER
# FINAL CONSISTENT VERSION
# ============================================================

print("=" * 70)
print("PERSON IDENTITY GALLERY BUILDER - FINAL")
print("=" * 70)


# ============================================================
# PATH CONFIGURATION
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

EMBEDDINGS_DIR = os.path.join(
    PERSON_IDENTITY_DIR,
    "embeddings"
)

DATABASE_PATH = os.path.join(
    PERSON_IDENTITY_DIR,
    "identity_database.db"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

REID_MODEL_NAME = "osnet_ibn_x1_0"

WEIGHTS_PATH = os.path.join(
    BASE_DIR,
    "weights",
    "osnet_ibn_x1_0_duke_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("")
print(f"Device: {DEVICE}")


# ============================================================
# SETTINGS
# ============================================================

YOLO_CONFIDENCE = 0.45
YOLO_IMAGE_SIZE = 640

# Expand detected person bounding box slightly.
BBOX_EXPANSION = 0.10

# Minimum acceptable detected person crop.
MIN_CROP_WIDTH = 40
MIN_CROP_HEIGHT = 80

# OSNet expected input size.
REID_WIDTH = 128
REID_HEIGHT = 256

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(
    EMBEDDINGS_DIR,
    exist_ok=True
)


# ============================================================
# CHECK OSNET WEIGHTS
# ============================================================

print("")
print("=" * 70)
print("CHECKING OSNET WEIGHTS")
print("=" * 70)

if not os.path.exists(
    WEIGHTS_PATH
):

    print("")
    print("ERROR: OSNet weights not found:")
    print(WEIGHTS_PATH)

    raise SystemExit(1)

print("Weights found:")
print(WEIGHTS_PATH)


# ============================================================
# LOAD YOLO
# ============================================================

print("")
print("=" * 70)
print("LOADING YOLO")
print("=" * 70)

try:

    yolo_model = YOLO(
        "yolo11n.pt"
    )

    print(
        "YOLO loaded successfully."
    )

except Exception as e:

    print(
        f"ERROR loading YOLO: {e}"
    )

    raise SystemExit(1)


# ============================================================
# LOAD OSNET
# ============================================================

print("")
print("=" * 70)
print("LOADING OSNET-IBN REID MODEL")
print("=" * 70)

try:

    print(
        f"Model: {REID_MODEL_NAME}"
    )

    reid_model = torchreid.models.build_model(
        name=REID_MODEL_NAME,
        num_classes=1000,
        pretrained=False
    )

    print(
        "Loading local ReID weights..."
    )

    torchreid.utils.load_pretrained_weights(
        reid_model,
        WEIGHTS_PATH
    )

    reid_model.eval()

    reid_model = reid_model.to(
        DEVICE
    )

    print(
        "OSNet-IBN loaded successfully."
    )

except Exception as e:

    print(
        f"ERROR loading OSNet: {e}"
    )

    raise SystemExit(1)


# ============================================================
# DATABASE
# ============================================================

def create_database():

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = connection.cursor()

    cursor.execute(
        """
        DROP TABLE IF EXISTS person_embeddings
        """
    )

    cursor.execute(
        """
        CREATE TABLE person_embeddings (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            person_id TEXT NOT NULL,

            image_name TEXT NOT NULL,

            embedding_path TEXT NOT NULL,

            embedding_dimension INTEGER NOT NULL,

            bbox_confidence REAL,

            bbox_x1 INTEGER,
            bbox_y1 INTEGER,
            bbox_x2 INTEGER,
            bbox_y2 INTEGER,

            crop_width INTEGER,
            crop_height INTEGER,

            preprocessing TEXT,

            created_at DATETIME
                DEFAULT CURRENT_TIMESTAMP

        )
        """
    )

    connection.commit()
    connection.close()

    print("")
    print("Database created successfully.")


# ============================================================
# CLEAR OLD EMBEDDINGS
# ============================================================

def clear_old_embeddings():

    print("")
    print("=" * 70)
    print("CLEARING OLD EMBEDDINGS")
    print("=" * 70)

    deleted = 0

    if not os.path.exists(
        EMBEDDINGS_DIR
    ):

        os.makedirs(
            EMBEDDINGS_DIR,
            exist_ok=True
        )

        return

    for filename in os.listdir(
        EMBEDDINGS_DIR
    ):

        if filename.lower().endswith(
            ".npy"
        ):

            path = os.path.join(
                EMBEDDINGS_DIR,
                filename
            )

            try:

                os.remove(
                    path
                )

                deleted += 1

            except Exception as e:

                print(
                    f"Could not delete "
                    f"{filename}: {e}"
                )

    print(
        f"Old embeddings removed: {deleted}"
    )


# ============================================================
# NORMALIZE EMBEDDING
# ============================================================

def normalize_embedding(
    embedding
):

    embedding = np.asarray(
        embedding,
        dtype=np.float32
    )

    norm = np.linalg.norm(
        embedding
    )

    if norm <= 0:

        return embedding

    return (
        embedding / norm
    )


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(
    a,
    b
):

    a = normalize_embedding(
        a
    )

    b = normalize_embedding(
        b
    )

    denominator = (
        np.linalg.norm(a)
        *
        np.linalg.norm(b)
    )

    if denominator <= 0:

        return 0.0

    return float(
        np.dot(a, b)
        /
        denominator
    )


# ============================================================
# EXPAND BOUNDING BOX
# ============================================================

def expand_bbox(
    coordinates,
    image_shape,
    expansion=BBOX_EXPANSION
):

    x1, y1, x2, y2 = map(
        float,
        coordinates
    )

    height, width = image_shape[:2]

    box_width = x2 - x1
    box_height = y2 - y1

    expand_x = (
        box_width * expansion
    )

    expand_y = (
        box_height * expansion
    )

    x1 -= expand_x
    y1 -= expand_y

    x2 += expand_x
    y2 += expand_y

    x1 = max(
        0,
        int(x1)
    )

    y1 = max(
        0,
        int(y1)
    )

    x2 = min(
        width,
        int(x2)
    )

    y2 = min(
        height,
        int(y2)
    )

    return (
        x1,
        y1,
        x2,
        y2
    )


# ============================================================
# CROP PERSON
# ============================================================

def crop_person(
    image,
    coordinates
):

    x1, y1, x2, y2 = expand_bbox(
        coordinates,
        image.shape
    )

    if (
        x2 <= x1
        or
        y2 <= y1
    ):

        return None, None

    crop = image[
        y1:y2,
        x1:x2
    ]

    if crop.size == 0:

        return None, None

    crop_height, crop_width = (
        crop.shape[:2]
    )

    if (
        crop_width < MIN_CROP_WIDTH
        or
        crop_height < MIN_CROP_HEIGHT
    ):

        print(
            f"  Crop too small: "
            f"{crop_width} x {crop_height}"
        )

        return None, None

    return (
        crop,
        (
            x1,
            y1,
            x2,
            y2
        )
    )


# ============================================================
# PREPROCESS IMAGE
#
# IMPORTANT:
# This preprocessing MUST match test_identity.py
# ============================================================

def preprocess_reid_image(
    person_image
):

    if person_image is None:

        return None

    # --------------------------------------------------------
    # BGR -> RGB
    # --------------------------------------------------------

    rgb = cv2.cvtColor(
        person_image,
        cv2.COLOR_BGR2RGB
    )

    pil_image = Image.fromarray(
        rgb
    )

    original_width, original_height = (
        pil_image.size
    )

    if (
        original_width <= 0
        or
        original_height <= 0
    ):

        return None

    target_width = REID_WIDTH
    target_height = REID_HEIGHT

    # --------------------------------------------------------
    # Preserve aspect ratio
    # --------------------------------------------------------

    scale = min(
        target_width / original_width,
        target_height / original_height
    )

    new_width = max(
        1,
        int(
            round(
                original_width * scale
            )
        )
    )

    new_height = max(
        1,
        int(
            round(
                original_height * scale
            )
        )
    )

    # --------------------------------------------------------
    # Resize
    # --------------------------------------------------------

    resized = pil_image.resize(
        (
            new_width,
            new_height
        ),
        Image.Resampling.BILINEAR
    )

    # --------------------------------------------------------
    # Padding
    # --------------------------------------------------------

    canvas = Image.new(
        "RGB",
        (
            target_width,
            target_height
        ),
        (
            0,
            0,
            0
        )
    )

    left = (
        target_width
        -
        new_width
    ) // 2

    top = (
        target_height
        -
        new_height
    ) // 2

    canvas.paste(
        resized,
        (
            left,
            top
        )
    )

    return canvas


# ============================================================
# SINGLE EMBEDDING
# ============================================================

def extract_single_embedding(
    person_image
):

    processed = preprocess_reid_image(
        person_image
    )

    if processed is None:

        return None

    image_array = (
        np.asarray(
            processed
        )
        .astype(
            np.float32
        )
        /
        255.0
    )

    # HWC -> CHW
    image_array = np.transpose(
        image_array,
        (
            2,
            0,
            1
        )
    )

    tensor = torch.from_numpy(
        image_array
    ).float()

    # --------------------------------------------------------
    # ImageNet normalization
    # --------------------------------------------------------

    mean = torch.tensor(
        [
            0.485,
            0.456,
            0.406
        ],
        dtype=torch.float32
    ).view(
        3,
        1,
        1
    )

    std = torch.tensor(
        [
            0.229,
            0.224,
            0.225
        ],
        dtype=torch.float32
    ).view(
        3,
        1,
        1
    )

    tensor = (
        tensor - mean
    ) / std

    tensor = tensor.unsqueeze(
        0
    )

    tensor = tensor.to(
        DEVICE
    )

    # --------------------------------------------------------
    # OSNet inference
    # --------------------------------------------------------

    with torch.no_grad():

        feature = reid_model(
            tensor
        )

    feature = (
        feature
        .detach()
        .cpu()
        .numpy()
        .flatten()
    )

    return normalize_embedding(
        feature
    )


# ============================================================
# ROBUST EMBEDDING
#
# ORIGINAL + HORIZONTAL FLIP
#
# MUST MATCH test_identity.py
# ============================================================

def extract_embedding(
    person_image
):

    if person_image is None:

        return None

    try:

        # ----------------------------------------------------
        # Original
        # ----------------------------------------------------

        original_embedding = (
            extract_single_embedding(
                person_image
            )
        )

        if original_embedding is None:

            return None

        # ----------------------------------------------------
        # Horizontal flip
        # ----------------------------------------------------

        flipped_image = cv2.flip(
            person_image,
            1
        )

        flipped_embedding = (
            extract_single_embedding(
                flipped_image
            )
        )

        if flipped_embedding is None:

            return original_embedding

        # ----------------------------------------------------
        # Average original + flip
        # ----------------------------------------------------

        embedding = (
            original_embedding
            +
            flipped_embedding
        ) / 2.0

        # ----------------------------------------------------
        # Final normalization
        # ----------------------------------------------------

        embedding = normalize_embedding(
            embedding
        )

        return embedding

    except Exception as e:

        print(
            f"  Embedding error: {e}"
        )

        return None


# ============================================================
# DETECT PERSON
# ============================================================

def detect_person(
    image
):

    results = yolo_model(
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

        return None

    # --------------------------------------------------------
    # Select highest confidence person
    # --------------------------------------------------------

    best_box, best_confidence = max(
        persons,
        key=lambda x: x[1]
    )

    crop, expanded_box = crop_person(
        image,
        best_box
    )

    if crop is None:

        return None

    return {
        "crop":
            crop,

        "confidence":
            best_confidence,

        "original_box":
            tuple(
                map(
                    int,
                    best_box
                )
            ),

        "expanded_box":
            expanded_box
    }


# ============================================================
# SAVE DATABASE RECORD
# ============================================================

def save_embedding_record(
    person_id,
    image_name,
    embedding_path,
    embedding_dimension,
    confidence,
    bbox,
    crop_width,
    crop_height
):

    connection = sqlite3.connect(
        DATABASE_PATH
    )

    cursor = connection.cursor()

    x1, y1, x2, y2 = bbox

    cursor.execute(
        """
        INSERT INTO person_embeddings (

            person_id,
            image_name,
            embedding_path,
            embedding_dimension,
            bbox_confidence,

            bbox_x1,
            bbox_y1,
            bbox_x2,
            bbox_y2,

            crop_width,
            crop_height,

            preprocessing

        )

        VALUES (
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?,
            ?, ?,
            ?
        )
        """,
        (
            person_id,
            image_name,
            embedding_path,
            embedding_dimension,
            confidence,

            x1,
            y1,
            x2,
            y2,

            crop_width,
            crop_height,

            "aspect_ratio_preserving_padding + flip_augmentation"
        )
    )

    connection.commit()
    connection.close()


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_image(
    person_id,
    image_path
):

    image_name = os.path.basename(
        image_path
    )

    print("")
    print(
        f"Processing: "
        f"{person_id} / {image_name}"
    )

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    image = cv2.imread(
        image_path
    )

    if image is None:

        print(
            "  ERROR: Could not read image."
        )

        return False

    # --------------------------------------------------------
    # Detect person
    # --------------------------------------------------------

    detection = detect_person(
        image
    )

    if detection is None:

        print(
            "  NO VALID PERSON DETECTED"
        )

        return False

    person_crop = detection[
        "crop"
    ]

    confidence = detection[
        "confidence"
    ]

    expanded_box = detection[
        "expanded_box"
    ]

    crop_height, crop_width = (
        person_crop.shape[:2]
    )

    print(
        f"  Person confidence: "
        f"{confidence:.3f}"
    )

    print(
        f"  Expanded crop: "
        f"{crop_width} x "
        f"{crop_height}"
    )

    # --------------------------------------------------------
    # Extract robust embedding
    # --------------------------------------------------------

    embedding = extract_embedding(
        person_crop
    )

    if embedding is None:

        print(
            "  ERROR: Embedding extraction failed."
        )

        return False

    print(
        f"  Embedding dimension: "
        f"{len(embedding)}"
    )

    # --------------------------------------------------------
    # Check embedding
    # --------------------------------------------------------

    embedding_norm = np.linalg.norm(
        embedding
    )

    if embedding_norm <= 0:

        print(
            "  ERROR: Invalid zero embedding."
        )

        return False

    # --------------------------------------------------------
    # Save embedding
    # --------------------------------------------------------

    base_name = os.path.splitext(
        image_name
    )[0]

    embedding_filename = (
        f"{person_id}__{base_name}.npy"
    )

    embedding_path = os.path.join(
        EMBEDDINGS_DIR,
        embedding_filename
    )

    np.save(
        embedding_path,
        embedding
    )

    print(
        f"  Saved: "
        f"{embedding_filename}"
    )

    # --------------------------------------------------------
    # Save database record
    # --------------------------------------------------------

    save_embedding_record(
        person_id,
        image_name,
        embedding_path,
        len(embedding),
        confidence,
        expanded_box,
        crop_width,
        crop_height
    )

    return True


# ============================================================
# BUILD GALLERY
# ============================================================

def build_gallery():

    print("")
    print("=" * 70)
    print("BUILDING PERSON IDENTITY GALLERY")
    print("=" * 70)

    if not os.path.exists(
        PERSONS_DIR
    ):

        print("")
        print(
            "ERROR: Persons directory does not exist:"
        )

        print(
            PERSONS_DIR
        )

        return False

    # --------------------------------------------------------
    # Find person folders
    # --------------------------------------------------------

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

    if not person_folders:

        print(
            "No person folders found."
        )

        return False

    person_folders.sort()

    print(
        f"People found: "
        f"{len(person_folders)}"
    )

    total_images = 0
    successful = 0
    failed = 0

    # --------------------------------------------------------
    # Process every person
    # --------------------------------------------------------

    for person_id in person_folders:

        person_directory = os.path.join(
            PERSONS_DIR,
            person_id
        )

        print("")
        print("-" * 70)
        print(
            f"PERSON: {person_id}"
        )
        print("-" * 70)

        images = []

        for filename in os.listdir(
            person_directory
        ):

            if filename.lower().endswith(
                IMAGE_EXTENSIONS
            ):

                images.append(
                    filename
                )

        images.sort()

        print(
            f"Images found: "
            f"{len(images)}"
        )

        if len(images) == 0:

            print(
                "WARNING: No images for this person."
            )

            continue

        # ----------------------------------------------------
        # Process images
        # ----------------------------------------------------

        for filename in images:

            total_images += 1

            image_path = os.path.join(
                person_directory,
                filename
            )

            success = process_image(
                person_id,
                image_path
            )

            if success:

                successful += 1

            else:

                failed += 1

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("")
    print("=" * 70)
    print("GALLERY BUILD COMPLETED")
    print("=" * 70)

    print(
        f"People processed : "
        f"{len(person_folders)}"
    )

    print(
        f"Images processed : "
        f"{total_images}"
    )

    print(
        f"Successful       : "
        f"{successful}"
    )

    print(
        f"Failed           : "
        f"{failed}"
    )

    print("")

    print(
        "Preprocessing:"
    )

    print(
        "1. YOLO person detection"
    )

    print(
        "2. 10% bounding-box expansion"
    )

    print(
        "3. Aspect-ratio preserving resize"
    )

    print(
        "4. Black padding"
    )

    print(
        "5. ImageNet normalization"
    )

    print(
        "6. Original + horizontal flip"
    )

    print(
        "7. Averaged embedding"
    )

    print(
        "8. L2 normalization"
    )

    print("")

    print(
        f"ReID input: "
        f"{REID_WIDTH} x {REID_HEIGHT}"
    )

    print(
        f"BBOX expansion: "
        f"{BBOX_EXPANSION * 100:.0f}%"
    )

    print("")

    print(
        "Embeddings directory:"
    )

    print(
        EMBEDDINGS_DIR
    )

    print("")

    print(
        "Database:"
    )

    print(
        DATABASE_PATH
    )

    return True


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("")
    print("=" * 70)
    print("STARTING GALLERY REBUILD")
    print("=" * 70)

    # --------------------------------------------------------
    # IMPORTANT:
    # Delete old embeddings FIRST
    # --------------------------------------------------------

    clear_old_embeddings()

    # --------------------------------------------------------
    # Recreate database
    # --------------------------------------------------------

    create_database()

    # --------------------------------------------------------
    # Build completely fresh gallery
    # --------------------------------------------------------

    success = build_gallery()

    print("")

    if success:

        print("=" * 70)
        print("SUCCESS: NEW GALLERY CREATED")
        print("=" * 70)

    else:

        print("=" * 70)
        print("GALLERY BUILD FAILED")
        print("=" * 70)