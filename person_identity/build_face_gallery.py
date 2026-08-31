"""
build_face_gallery.py

Builds a FACE-embedding gallery from the same folder structure you already
use for the ReID gallery:

    person_identity/persons/person_001/*.jpg
    person_identity/persons/person_002/*.jpg
    ...

Why face recognition instead of body-ReID:
Your reference photos show the same person in different outfits, poses,
and settings (casual/social photos, not same-session CCTV crops). Body
appearance ReID models (OSNet etc.) rely on clothing consistency and will
NOT separate identities in this situation. Face embeddings are clothing-
independent and are the right tool here.

Requirements:
    pip install insightface onnxruntime opencv-python numpy

Run:
    python build_face_gallery.py
"""

import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

# ============================================================
# PATHS  (mirrors your existing person_identity folder layout)
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PERSONS_DIR = os.path.join(BASE_DIR, "persons")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "embeddings_faces")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# ============================================================
# LOAD FACE MODEL
# ============================================================

print("=" * 70)
print("FACE GALLERY BUILDER")
print("=" * 70)
print("")
print("Loading InsightFace (buffalo_l)...")

face_app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
face_app.prepare(ctx_id=0, det_size=(640, 640))

print("Loaded successfully.")


def get_embedding(image):
    """Detect faces, return the most confident face's normalized embedding."""
    faces = face_app.get(image)
    if not faces:
        return None, 0

    faces.sort(key=lambda f: f.det_score, reverse=True)
    best = faces[0]

    emb = getattr(best, "normed_embedding", None)
    if emb is None:
        emb = best.embedding / np.linalg.norm(best.embedding)

    return emb.astype(np.float32), len(faces)


def main():
    if not os.path.exists(PERSONS_DIR):
        print(f"\nERROR: {PERSONS_DIR} does not exist.")
        return

    person_ids = sorted(
        d for d in os.listdir(PERSONS_DIR)
        if os.path.isdir(os.path.join(PERSONS_DIR, d))
    )

    print(f"\nPeople found: {len(person_ids)}")

    total, saved, skipped = 0, 0, 0

    for person_id in person_ids:
        person_dir = os.path.join(PERSONS_DIR, person_id)
        print("")
        print("-" * 70)
        print(f"PERSON: {person_id}")
        print("-" * 70)

        for filename in sorted(os.listdir(person_dir)):
            if not filename.lower().endswith(IMAGE_EXTENSIONS):
                continue

            total += 1
            path = os.path.join(person_dir, filename)
            image = cv2.imread(path)

            if image is None:
                print(f"  [{filename}] could not read image -> skipped")
                skipped += 1
                continue

            embedding, num_faces = get_embedding(image)

            if embedding is None:
                print(f"  [{filename}] NO FACE DETECTED -> skipped "
                      f"(use a clearer, front-facing photo)")
                skipped += 1
                continue

            if num_faces > 1:
                print(f"  [{filename}] {num_faces} faces found, "
                      f"used the most confident detection")

            base = os.path.splitext(filename)[0]
            out_path = os.path.join(EMBEDDINGS_DIR, f"{person_id}__{base}.npy")
            np.save(out_path, embedding)

            print(f"  [{filename}] OK -> saved {os.path.basename(out_path)}")
            saved += 1

    print("")
    print("=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Total images : {total}")
    print(f"Saved        : {saved}")
    print(f"Skipped      : {skipped}")
    print(f"Embeddings   : {EMBEDDINGS_DIR}")

    if skipped > 0:
        print("")
        print("NOTE: Images where no face was detected were skipped.")
        print("Replace those with clearer front-facing photos for best accuracy.")


if __name__ == "__main__":
    main()
