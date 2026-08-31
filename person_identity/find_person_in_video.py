"""
find_person_in_video.py

THE MAIN SCRIPT YOU ASKED FOR.

Give it ONE reference photo of a person + a video (from some camera).
It answers:

  - If the person appears in the video:
        "FOUND -> first/last seen at <time>, camera <X>"
  - If the person does NOT appear in this video:
        it checks the sightings database (built from all videos you've
        ever scanned) and tells you the last camera/time they WERE seen,
        or says there is no record at all.

Every match is logged to a small SQLite database (identity_database.db)
so future queries can answer "when was this person last seen anywhere".

Requirements:
    pip install insightface onnxruntime opencv-python numpy

Example:
    python find_person_in_video.py ^
        --query "D:\path\to\photo.jpg" ^
        --video "D:\path\to\camera1.mp4" ^
        --camera "Camera 1 - Main Gate" ^
        --person-id person_004

Optional flags:
    --interval 1.0        seconds between analyzed frames (default 1.0)
    --threshold 0.38       match strictness, 0-1, higher = stricter (default 0.38)
    --video-start "2026-08-20 10:00:00"   real clock time the video started

NOTE on --threshold:
    0.38 is a reasonable starting point for InsightFace buffalo_l cosine
    similarity. If you get false matches, raise it (e.g. 0.45). If a
    real match is being missed, lower it slightly (e.g. 0.32). Test on
    a known video first before trusting it for real decisions.
"""

import os
import sqlite3
import argparse
import datetime

import cv2
import numpy as np
from insightface.app import FaceAnalysis

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "identity_database.db")

DEFAULT_THRESHOLD = 0.38
DEFAULT_INTERVAL = 1.0


# ============================================================
# DATABASE
# ============================================================

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sightings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id TEXT NOT NULL,
            camera TEXT,
            video_source TEXT,
            frame_number INTEGER,
            seconds_into_video REAL,
            real_datetime TEXT,
            similarity REAL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn


def log_sighting(conn, person_id, camera, video_source, frame_number,
                  seconds, real_datetime, similarity):
    conn.execute("""
        INSERT INTO sightings
            (person_id, camera, video_source, frame_number,
             seconds_into_video, real_datetime, similarity)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (person_id, camera, video_source, frame_number,
          seconds, real_datetime, similarity))
    conn.commit()


def get_last_known_sighting(conn, person_id):
    cursor = conn.execute("""
        SELECT camera, video_source, seconds_into_video, real_datetime,
               similarity, created_at
        FROM sightings
        WHERE person_id = ?
        ORDER BY created_at DESC
        LIMIT 1
    """, (person_id,))
    return cursor.fetchone()


# ============================================================
# FACE MODEL
# ============================================================

def load_face_app():
    print("Loading InsightFace (buffalo_l)...")
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("Loaded.")
    return app


def get_embedding(face_app, image):
    faces = face_app.get(image)
    if not faces:
        return None
    faces.sort(key=lambda f: f.det_score, reverse=True)
    best = faces[0]
    emb = getattr(best, "normed_embedding", None)
    if emb is None:
        emb = best.embedding / np.linalg.norm(best.embedding)
    return emb.astype(np.float32)


def cosine(a, b):
    return float(np.dot(a, b))


def format_time(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))


# ============================================================
# VIDEO SCAN
# ============================================================

def scan_video(face_app, video_path, query_embedding, interval, threshold):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_step = max(1, int(round(fps * interval)))

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
                    if score >= threshold:
                        seconds = frame_number / fps
                        sightings.append((frame_number, seconds, score))

        frame_number += 1

    cap.release()
    return sightings, fps


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True,
                         help="Path to a reference photo of the person")
    parser.add_argument("--video", required=True,
                         help="Path to the video to search")
    parser.add_argument("--camera", required=True,
                         help="Camera name/id this video belongs to")
    parser.add_argument("--person-id", required=True,
                         help="Consistent id/name for this person, e.g. person_004")
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--video-start", default=None,
                         help='Real start time of the video, e.g. "2026-08-20 10:00:00"')
    args = parser.parse_args()

    if not os.path.exists(args.query):
        print(f"ERROR: query image not found: {args.query}")
        return
    if not os.path.exists(args.video):
        print(f"ERROR: video not found: {args.video}")
        return

    face_app = load_face_app()
    conn = get_db()

    query_image = cv2.imread(args.query)
    if query_image is None:
        print("ERROR: could not read query image.")
        return

    query_embedding = get_embedding(face_app, query_image)
    if query_embedding is None:
        print("ERROR: no face detected in the query photo.")
        print("Use a clearer, front-facing photo.")
        return

    print("")
    print(f"Searching for '{args.person_id}' in: {args.video}")
    print(f"Camera: {args.camera}   Threshold: {args.threshold}   "
          f"Interval: {args.interval}s")
    print("")

    sightings, fps = scan_video(
        face_app, args.video, query_embedding, args.interval, args.threshold
    )

    video_start_dt = None
    if args.video_start:
        video_start_dt = datetime.datetime.strptime(
            args.video_start, "%Y-%m-%d %H:%M:%S"
        )

    if sightings:
        first_frame, first_seconds, first_score = sightings[0]
        last_frame, last_seconds, last_score = sightings[-1]

        first_real = (video_start_dt + datetime.timedelta(seconds=first_seconds)) \
            if video_start_dt else None
        last_real = (video_start_dt + datetime.timedelta(seconds=last_seconds)) \
            if video_start_dt else None

        for frame_number, seconds, score in sightings:
            real_dt = (video_start_dt + datetime.timedelta(seconds=seconds)) \
                if video_start_dt else None
            log_sighting(
                conn, args.person_id, args.camera, args.video,
                frame_number, seconds,
                real_dt.strftime("%Y-%m-%d %H:%M:%S") if real_dt else None,
                score
            )

        print("RESULT: PERSON FOUND in this video.")
        print(f"  First seen : {format_time(first_seconds)} into the video "
              f"(similarity {first_score:.3f})"
              + (f", real time {first_real}" if first_real else ""))
        print(f"  Last seen  : {format_time(last_seconds)} into the video "
              f"(similarity {last_score:.3f})"
              + (f", real time {last_real}" if last_real else ""))
        print(f"  Camera     : {args.camera}")
        print(f"  Matched frames: {len(sightings)}")

    else:
        print("RESULT: Person NOT found in this video.")
        print("")
        print("Checking previous records...")

        last = get_last_known_sighting(conn, args.person_id)
        if last:
            camera, video_source, seconds, real_dt, similarity, created_at = last
            print(f"  Last known sighting -> camera: {camera}")
            print(f"  Video source: {video_source}")
            if real_dt:
                print(f"  Time: {real_dt}")
            else:
                print(f"  Time into that video: {format_time(seconds)}")
            print(f"  Similarity at the time: {similarity:.3f}")
            print(f"  Logged at: {created_at}")
        else:
            print("  No previous sighting record found for this person "
                  "in any camera.")

    conn.close()


if __name__ == "__main__":
    main()
