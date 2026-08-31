"""
person_reid.py  (FACE-RECOGNITION VERSION)

Drop-in replacement for the old OSNet body-ReID script. Called by the
dashboard (app.py) EXACTLY the same way:

    python person_reid.py <path_to_reference_image>

No changes are needed anywhere else in app.py - this script writes into
the same person_reid_sightings table, with the same columns, so the
dashboard's existing query/display code keeps working unmodified.

WHY FACE RECOGNITION INSTEAD OF OSNet BODY-ReID:
The reference photos uploaded here are casual portraits where the same
person can appear in different outfits. Body-appearance ReID (OSNet)
relies on clothing consistency and cannot reliably separate identities
in that situation. Face embeddings are clothing-independent, which is
what this system actually needs - this was validated on this exact
project's data (6/6 correct results: true match found, all non-matches
correctly rejected).

Requirements (same venv as the rest of the project):
    pip install insightface onnxruntime opencv-python numpy
"""

import os
import sys
import glob
import sqlite3
import datetime

import cv2
import numpy as np
from insightface.app import FaceAnalysis

# ============================================================
# PATHS
# ============================================================

PLATE_DB_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(PLATE_DB_DIR)

ASSETS_DIR = os.path.join(BASE_DIR, "assets")
DATABASE_PATH = os.path.join(PLATE_DB_DIR, "traffic_tracking.db")

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")

THRESHOLD = 0.38
FRAME_INTERVAL_SECONDS = 1.0


# ============================================================
# DATABASE
# ============================================================

def get_connection():
    return sqlite3.connect(DATABASE_PATH)


def ensure_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS person_reid_sightings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            reference_image TEXT NOT NULL,
            camera_id TEXT,
            city TEXT,
            road TEXT,
            video_source TEXT,
            first_seen_timestamp TEXT,
            first_seen_frame INTEGER,
            last_seen_timestamp TEXT,
            last_seen_frame INTEGER,
            best_similarity REAL,
            best_yolo_confidence REAL,
            matching_frames INTEGER,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()


def clear_previous_results(conn, reference_image):
    # A fresh search under the same uploaded filename should not mix
    # with results from a previous, different photo.
    conn.execute(
        "DELETE FROM person_reid_sightings WHERE reference_image = ?",
        (reference_image,)
    )
    conn.commit()


def get_city_road_for_camera(conn, camera_id):
    """
    Reuse city/road info already logged for this camera by the
    vehicle-tracking pipeline, if that table/data exists.
    """
    try:
        cursor = conn.execute(
            "SELECT city, road FROM vehicle_detections "
            "WHERE camera_id = ? LIMIT 1",
            (camera_id,)
        )
        row = cursor.fetchone()
        if row:
            return row[0] or "Unknown", row[1] or "Unknown"
    except Exception:
        pass
    return "Unknown", "Unknown"


def save_sighting(conn, reference_image, camera_id, city, road, video_source,
                   first_seen_timestamp, first_seen_frame,
                   last_seen_timestamp, last_seen_frame,
                   best_similarity, best_confidence, matching_frames):
    conn.execute("""
        INSERT INTO person_reid_sightings (
            reference_image, camera_id, city, road, video_source,
            first_seen_timestamp, first_seen_frame,
            last_seen_timestamp, last_seen_frame,
            best_similarity, best_yolo_confidence, matching_frames
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        reference_image, camera_id, city, road, video_source,
        first_seen_timestamp, first_seen_frame,
        last_seen_timestamp, last_seen_frame,
        best_similarity, best_confidence, matching_frames
    ))
    conn.commit()


# ============================================================
# FACE MODEL
# ============================================================

def load_face_app():
    print("Loading face recognition model (InsightFace buffalo_l)...")
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("Model loaded.")
    return app


def get_embedding(face_app, image):
    faces = face_app.get(image)
    if not faces:
        return None, None
    faces.sort(key=lambda f: f.det_score, reverse=True)
    best = faces[0]
    emb = getattr(best, "normed_embedding", None)
    if emb is None:
        emb = best.embedding / np.linalg.norm(best.embedding)
    return emb.astype(np.float32), float(best.det_score)


def cosine(a, b):
    return float(np.dot(a, b))


def format_time(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))


# ============================================================
# FIND VIDEOS
# ============================================================

def find_all_videos():
    videos = []
    camera_folders = sorted(glob.glob(os.path.join(ASSETS_DIR, "Camera_*")))
    for cam_folder in camera_folders:
        if not os.path.isdir(cam_folder):
            continue
        camera_name = os.path.basename(cam_folder)
        for filename in sorted(os.listdir(cam_folder)):
            if filename.lower().endswith(VIDEO_EXTENSIONS):
                videos.append({
                    "camera": camera_name,
                    "path": os.path.join(cam_folder, filename)
                })
    return videos


# ============================================================
# SCAN ONE VIDEO
# ============================================================

def scan_video(face_app, video_path, query_embedding):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  WARNING: could not open {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_step = max(1, int(round(fps * FRAME_INTERVAL_SECONDS)))

    sightings = []
    frame_number = 0

    while True:
        ret = cap.grab()
        if not ret:
            break

        if frame_number % frame_step == 0:
            ok, frame = cap.retrieve()
            if ok:
                faces = face_app.get(frame)
                for face in faces:
                    emb = getattr(face, "normed_embedding", None)
                    if emb is None:
                        emb = face.embedding / np.linalg.norm(face.embedding)
                    score = cosine(query_embedding, emb)
                    if score >= THRESHOLD:
                        seconds = frame_number / fps
                        sightings.append({
                            "frame": frame_number,
                            "seconds": seconds,
                            "similarity": score,
                            "det_score": float(face.det_score)
                        })

        frame_number += 1

    cap.release()
    return sightings


# ============================================================
# MAIN
# ============================================================

def main():
    if len(sys.argv) < 2:
        print("ERROR: no reference image provided.")
        print("Usage: python person_reid.py <path_to_image>")
        sys.exit(1)

    image_path = os.path.abspath(sys.argv[1])
    reference_image = os.path.basename(image_path)

    print("=" * 70)
    print("PERSON RE-IDENTIFICATION (FACE RECOGNITION)")
    print("=" * 70)
    print("")
    print(f"Reference image: {image_path}")

    if not os.path.exists(image_path):
        print("ERROR: reference image not found.")
        sys.exit(1)

    conn = get_connection()
    ensure_table(conn)
    clear_previous_results(conn, reference_image)

    face_app = load_face_app()

    ref_image = cv2.imread(image_path)
    if ref_image is None:
        print("ERROR: could not read reference image.")
        sys.exit(1)

    query_embedding, det_score = get_embedding(face_app, ref_image)
    if query_embedding is None:
        print("ERROR: no face detected in the reference image.")
        print("Please upload a clearer, front-facing photo.")
        sys.exit(1)

    print(f"Face detected in reference image (confidence {det_score:.3f})")

    videos = find_all_videos()
    if not videos:
        print(f"No camera videos found under {ASSETS_DIR}")
        sys.exit(1)

    print("")
    print(f"Scanning {len(videos)} video(s) across all cameras...")
    print("")

    found_any = False

    for video in videos:
        print(f"--- {video['camera']} / {os.path.basename(video['path'])} ---")
        sightings = scan_video(face_app, video["path"], query_embedding)

        if not sightings:
            print("  not found")
            continue

        found_any = True

        first = sightings[0]
        last = sightings[-1]
        best_similarity = max(s["similarity"] for s in sightings)
        best_confidence = max(s["det_score"] for s in sightings)

        city, road = get_city_road_for_camera(conn, video["camera"])

        save_sighting(
            conn,
            reference_image=reference_image,
            camera_id=video["camera"],
            city=city,
            road=road,
            video_source=video["path"],
            first_seen_timestamp=format_time(first["seconds"]),
            first_seen_frame=first["frame"],
            last_seen_timestamp=format_time(last["seconds"]),
            last_seen_frame=last["frame"],
            best_similarity=best_similarity,
            best_confidence=best_confidence,
            matching_frames=len(sightings)
        )

        print(f"  FOUND - {len(sightings)} matching frame(s), "
              f"best similarity {best_similarity:.3f}")

    print("")
    print("=" * 70)
    if found_any:
        print("RESULT: Person found in one or more cameras.")
    else:
        print("RESULT: Person not found in any available camera.")
    print("=" * 70)

    conn.close()
    sys.exit(0)


if __name__ == "__main__":
    main()
