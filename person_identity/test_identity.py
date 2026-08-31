import os
import sys
import cv2
import numpy as np
import torch
import torchreid

from ultralytics import YOLO
from PIL import Image


# ============================================================
# PERSON IDENTITY TEST
# ============================================================

print("=" * 70)
print("PERSON IDENTITY RECOGNITION TEST - V3")
print("=" * 70)


# ============================================================
# PATHS
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

EMBEDDINGS_DIR = os.path.join(
    PERSON_IDENTITY_DIR,
    "embeddings"
)


# ============================================================
# MODEL
# ============================================================

REID_MODEL_NAME = "osnet_ibn_x1_0"

WEIGHTS_PATH = os.path.join(
    BASE_DIR,
    "weights",
    "osnet_ibn_x1_0_duke_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip.pth"
)


# ============================================================
# SETTINGS
# ============================================================

YOLO_CONFIDENCE = 0.45
YOLO_IMAGE_SIZE = 640

BBOX_EXPANSION = 0.10

REID_WIDTH = 128
REID_HEIGHT = 256

MIN_CROP_WIDTH = 40
MIN_CROP_HEIGHT = 80


# ============================================================
# MATCH SETTINGS
# ============================================================

MATCH_THRESHOLD = 0.68

SUPPORT_THRESHOLD = 0.55

SUPPORT_COUNT = 2

MIN_MARGIN = 0.04

TOP_K = 3


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
# INPUT
# ============================================================

if len(sys.argv) < 2:

    print("")
    print("ERROR: No test image provided.")

    print("")
    print("Usage:")

    print(
        'python person_identity\\test_identity.py "path_to_image"'
    )

    sys.exit(1)


TEST_IMAGE = os.path.abspath(
    sys.argv[1]
)

print("")
print("Test image:")
print(TEST_IMAGE)


if not os.path.exists(TEST_IMAGE):

    print("")
    print("ERROR: Test image does not exist.")

    sys.exit(1)


# ============================================================
# CHECK WEIGHTS
# ============================================================

print("")
print("=" * 70)
print("CHECKING OSNET WEIGHTS")
print("=" * 70)

if not os.path.exists(WEIGHTS_PATH):

    print("")
    print("ERROR: OSNet weights not found.")
    print(WEIGHTS_PATH)

    sys.exit(1)

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

    sys.exit(1)


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
        "OSNet-IBN ReID model loaded successfully."
    )

except Exception as e:

    print(
        f"ERROR loading OSNet: {e}"
    )

    sys.exit(1)


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(a, b):

    a = np.asarray(
        a,
        dtype=np.float32
    )

    b = np.asarray(
        b,
        dtype=np.float32
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
# NORMALIZE
# ============================================================

def normalize_embedding(embedding):

    embedding = np.asarray(
        embedding,
        dtype=np.float32
    )

    norm = np.linalg.norm(
        embedding
    )

    if norm <= 0:

        return embedding

    return embedding / norm


# ============================================================
# PREPROCESS
#
# IMPORTANT:
# THIS IS NOW IDENTICAL TO build_gallery.py
# ============================================================

def preprocess_reid_image(person_image):

    if person_image is None:

        return None

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

    scale = min(
        REID_WIDTH / original_width,
        REID_HEIGHT / original_height
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

    resized = pil_image.resize(
        (
            new_width,
            new_height
        ),
        Image.Resampling.BILINEAR
    )

    canvas = Image.new(
        "RGB",
        (
            REID_WIDTH,
            REID_HEIGHT
        ),
        (
            0,
            0,
            0
        )
    )

    left = (
        REID_WIDTH
        -
        new_width
    ) // 2

    top = (
        REID_HEIGHT
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

def extract_single_embedding(person_image):

    image = preprocess_reid_image(
        person_image
    )

    if image is None:

        return None

    image = (
        np.asarray(
            image
        )
        .astype(
            np.float32
        )
        / 255.0
    )

    image = np.transpose(
        image,
        (
            2,
            0,
            1
        )
    )

    tensor = torch.from_numpy(
        image
    ).float()

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
# SAME STRATEGY USED FOR BOTH
# GALLERY + QUERY
# ============================================================

def extract_embedding(person_image):

    if person_image is None:

        return None

    if (
        person_image.shape[1]
        <
        MIN_CROP_WIDTH
        or
        person_image.shape[0]
        <
        MIN_CROP_HEIGHT
    ):

        print(
            "ERROR: Person crop is too small."
        )

        return None

    try:

        original_embedding = (
            extract_single_embedding(
                person_image
            )
        )

        flipped_image = cv2.flip(
            person_image,
            1
        )

        flipped_embedding = (
            extract_single_embedding(
                flipped_image
            )
        )

        if (
            original_embedding is None
            or
            flipped_embedding is None
        ):

            return None

        embedding = (
            original_embedding
            +
            flipped_embedding
        ) / 2.0

        return normalize_embedding(
            embedding
        )

    except Exception as e:

        print(
            f"Embedding error: {e}"
        )

        return None


# ============================================================
# BBOX
# ============================================================

def expand_bbox(
    image,
    coordinates,
    expansion=BBOX_EXPANSION
):

    x1, y1, x2, y2 = map(
        float,
        coordinates
    )

    height, width = image.shape[:2]

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
# DETECT PERSON
# ============================================================

def detect_person(image):

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

    best_box, confidence = max(
        persons,
        key=lambda x: x[1]
    )

    x1, y1, x2, y2 = expand_bbox(
        image,
        best_box
    )

    if (
        x2 <= x1
        or
        y2 <= y1
    ):

        return None

    crop = image[
        y1:y2,
        x1:x2
    ]

    if crop.size == 0:

        return None

    return (
        crop,
        confidence,
        (
            x1,
            y1,
            x2,
            y2
        )
    )


# ============================================================
# LOAD GALLERY
# ============================================================

def load_gallery():

    print("")
    print("=" * 70)
    print("LOADING IDENTITY GALLERY")
    print("=" * 70)

    if not os.path.exists(
        EMBEDDINGS_DIR
    ):

        print(
            "ERROR: Embeddings directory not found."
        )

        return {}

    gallery = {}

    for filename in sorted(
        os.listdir(
            EMBEDDINGS_DIR
        )
    ):

        if not filename.lower().endswith(
            ".npy"
        ):

            continue

        path = os.path.join(
            EMBEDDINGS_DIR,
            filename
        )

        try:

            embedding = np.load(
                path
            )

            embedding = normalize_embedding(
                embedding
            )

            person_id = filename.split(
                "__"
            )[0]

            gallery.setdefault(
                person_id,
                []
            )

            gallery[person_id].append(
                {
                    "filename": filename,
                    "embedding": embedding
                }
            )

        except Exception as e:

            print(
                f"Could not load {filename}: {e}"
            )

    for person_id in sorted(
        gallery
    ):

        print(
            f"{person_id}: "
            f"{len(gallery[person_id])} embeddings"
        )

    print("")
    print(
        f"Total identities: "
        f"{len(gallery)}"
    )

    return gallery


# ============================================================
# COMPARE
# ============================================================

def compare_identity(
    query_embedding,
    person_gallery
):

    comparisons = []

    for item in person_gallery:

        similarity = cosine_similarity(
            query_embedding,
            item["embedding"]
        )

        comparisons.append(
            {
                "filename":
                    item["filename"],

                "similarity":
                    similarity
            }
        )

    comparisons.sort(
        key=lambda x:
        x["similarity"],
        reverse=True
    )

    similarities = [
        x["similarity"]
        for x in comparisons
    ]

    best_similarity = similarities[0]

    top_k = similarities[
        :min(
            TOP_K,
            len(similarities)
        )
    ]

    top_average = float(
        np.mean(top_k)
    )

    top_median = float(
        np.median(top_k)
    )

    support_count = sum(
        1
        for similarity in similarities
        if similarity >= SUPPORT_THRESHOLD
    )

    identity_score = (
        0.40 * best_similarity
        +
        0.35 * top_average
        +
        0.25 * top_median
    )

    return {
        "best_similarity":
            best_similarity,

        "top_average":
            top_average,

        "top_median":
            top_median,

        "support_count":
            support_count,

        "identity_score":
            identity_score,

        "comparisons":
            comparisons
    }


# ============================================================
# SEARCH
# ============================================================

def search_gallery(
    query_embedding,
    gallery
):

    results = []

    for person_id, person_gallery in (
        gallery.items()
    ):

        result = compare_identity(
            query_embedding,
            person_gallery
        )

        result["person_id"] = person_id

        results.append(
            result
        )

    results.sort(
        key=lambda x:
        x["identity_score"],
        reverse=True
    )

    return results


# ============================================================
# MAIN
# ============================================================

def main():

    gallery = load_gallery()

    if not gallery:

        print(
            "\nERROR: Gallery is empty."
        )

        return 1

    print("")
    print("=" * 70)
    print("READING TEST IMAGE")
    print("=" * 70)

    image = cv2.imread(
        TEST_IMAGE
    )

    if image is None:

        print(
            "ERROR: Could not read test image."
        )

        return 1

    print(
        "Test image loaded."
    )

    print("")
    print(
        "Detecting person..."
    )

    detection = detect_person(
        image
    )

    if detection is None:

        print(
            "\nNO PERSON DETECTED."
        )

        return 1

    (
        person_crop,
        confidence,
        bbox
    ) = detection

    print(
        f"YOLO confidence: "
        f"{confidence:.3f}"
    )

    print(
        f"Expanded crop: "
        f"{person_crop.shape[1]} x "
        f"{person_crop.shape[0]}"
    )

    print("")
    print(
        "Extracting OSNet-IBN embedding..."
    )

    query_embedding = extract_embedding(
        person_crop
    )

    if query_embedding is None:

        print(
            "ERROR: Could not create embedding."
        )

        return 1

    print(
        f"Query embedding dimension: "
        f"{len(query_embedding)}"
    )

    print("")
    print("=" * 70)
    print("SEARCHING IDENTITY GALLERY")
    print("=" * 70)

    results = search_gallery(
        query_embedding,
        gallery
    )

    if not results:

        print(
            "No identities found."
        )

        return 1

    print("")

    for index, result in enumerate(
        results,
        start=1
    ):

        print(
            f"{index}. "
            f"{result['person_id']:<15}"
            f"Identity Score: "
            f"{result['identity_score']:.4f}   "
            f"Best: "
            f"{result['best_similarity']:.4f}   "
            f"Top-{TOP_K} Avg: "
            f"{result['top_average']:.4f}   "
            f"Support: "
            f"{result['support_count']}"
        )

    best = results[0]

    second_best = (
        results[1]
        if len(results) > 1
        else None
    )

    if second_best:

        margin = (
            best["identity_score"]
            -
            second_best["identity_score"]
        )

    else:

        margin = 1.0

    print("")
    print("=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    print("")
    print(
        f"Best identity: "
        f"{best['person_id']}"
    )

    print(
        f"Identity score: "
        f"{best['identity_score']:.4f}"
    )

    print(
        f"Best similarity: "
        f"{best['best_similarity']:.4f}"
    )

    print(
        f"Top-{TOP_K} average: "
        f"{best['top_average']:.4f}"
    )

    print(
        f"Supporting images: "
        f"{best['support_count']}"
    )

    print(
        f"Margin over second identity: "
        f"{margin:.4f}"
    )

    strong_score = (
        best["identity_score"]
        >= MATCH_THRESHOLD
    )

    enough_support = (
        best["support_count"]
        >= SUPPORT_COUNT
    )

    sufficient_margin = (
        margin
        >= MIN_MARGIN
    )

    if (
        strong_score
        and
        enough_support
        and
        sufficient_margin
    ):

        print("")
        print(
            "STATUS: MATCH FOUND"
        )

        print(
            f"Identity: "
            f"{best['person_id']}"
        )

    else:

        print("")
        print(
            "STATUS: UNKNOWN / INSUFFICIENT EVIDENCE"
        )

        if not strong_score:

            print(
                "- Identity score is too low."
            )

        if not enough_support:

            print(
                "- Not enough supporting gallery images."
            )

        if not sufficient_margin:

            print(
                "- Difference from second identity "
                "is too small."
            )

    print("")
    print("=" * 70)

    return 0


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    sys.exit(
        main()
    )