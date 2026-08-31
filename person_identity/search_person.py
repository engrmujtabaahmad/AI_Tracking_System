r"""
search_person.py

FINAL / PRODUCTION ENTRY POINT.

Give it ONE query photo. It automatically:
  1. Scans every video inside every Camera_* folder under assets\
  2. Reports every sighting (camera + time)
  3. If not found anywhere in the current scan, checks the sightings
     database for the most recent PREVIOUS sighting of this person
  4. Saves a full report to person_identity/reports/<person_id>_<timestamp>.json
     so it can be plugged straight into your dashboard/backend.

Usage:
    python search_person.py --query "path\to\photo.jpg" --person-id person_004

Optional:
    --assets-dir "D:\AI_Tracking_System\assets"   (default below)
    --interval 1.0
    --threshold 0.38
    --video-start "2026-08-20 10:00:00"   (applies to ALL videos scanned - only
                                            use this if every video started at
                                            the same real-world time)
"""

import os
import glob
import json
import sqlite3
import argparse
import datetime

import cv2
import numpy as np
from insightface.app import FaceAnalysis

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ASSETS_DIR = os.path.join(os.path.dirname(BASE_DIR), "assets")
DB_PATH = os.path.join(BASE_DIR, "identity_database.db")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")

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
# FIND ALL VIDEOS
# ============================================================

def find_all_videos(assets_dir):
    videos = []
    camera_folders = sorted(glob.glob(os.path.join(assets_dir, "Camera_*")))
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

def scan_video(face_app, video_path, query_embedding, interval, threshold):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  WARNING: could not open {video_path}, skipping.")
        return [], 0

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
    parser.add_argument("--person-id", required=True,
                         help="Consistent id/name for this person, e.g. person_004")
    parser.add_argument("--assets-dir", default=DEFAULT_ASSETS_DIR)
    parser.add_argument("--interval", type=float, default=DEFAULT_INTERVAL)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--video-start", default=None)
    args = parser.parse_args()

    if not os.path.exists(args.query):
        print(f"ERROR: query image not found: {args.query}")
        return

    os.makedirs(REPORTS_DIR, exist_ok=True)

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

    videos = find_all_videos(args.assets_dir)
    if not videos:
        print(f"No videos found under {args.assets_dir} (Camera_* folders).")
        return

    video_start_dt = None
    if args.video_start:
        video_start_dt = datetime.datetime.strptime(
            args.video_start, "%Y-%m-%d %H:%M:%S"
        )

    print("")
    print(f"Searching for '{args.person_id}' across {len(videos)} video(s)...")
    print("")

    all_matches = []

    for video in videos:
        print(f"--- {video['camera']} / {os.path.basename(video['path'])} ---")
        sightings, fps = scan_video(
            face_app, video["path"], query_embedding,
            args.interval, args.threshold
        )

        if sightings:
            print(f"  FOUND - {len(sightings)} matching frame(s)")
            for frame_number, seconds, score in sightings:
                real_dt = (video_start_dt + datetime.timedelta(seconds=seconds)) \
                    if video_start_dt else None
                real_dt_str = real_dt.strftime("%Y-%m-%d %H:%M:%S") if real_dt else None

                log_sighting(
                    conn, args.person_id, video["camera"], video["path"],
                    frame_number, seconds, real_dt_str, score
                )

                all_matches.append({
                    "camera": video["camera"],
                    "video": video["path"],
                    "frame_number": frame_number,
                    "seconds_into_video": round(seconds, 2),
                    "time_into_video": format_time(seconds),
                    "real_datetime": real_dt_str,
                    "similarity": round(score, 4)
                })
        else:
            print("  not found")

    print("")
    print("=" * 70)
    print("FINAL REPORT")
    print("=" * 70)

    report = {
        "person_id": args.person_id,
        "query_photo": args.query,
        "scanned_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "videos_scanned": len(videos),
        "threshold": args.threshold,
        "found_in_current_scan": len(all_matches) > 0,
        "matches": all_matches,
        "last_known_sighting_previous": None
    }

    if all_matches:
        best = max(all_matches, key=lambda m: m["similarity"])
        print(f"STATUS: FOUND")
        print(f"  Best match : {best['camera']} at {best['time_into_video']} "
              f"(similarity {best['similarity']})")
        print(f"  Total sightings this scan: {len(all_matches)}")
    else:
        print("STATUS: NOT FOUND in any scanned video.")
        last = get_last_known_sighting(conn, args.person_id)
        if last:
            camera, video_source, seconds, real_dt, similarity, created_at = last
            print("  Last known sighting (from previous scans):")
            print(f"    Camera   : {camera}")
            print(f"    Video    : {video_source}")
            print(f"    Time     : {real_dt if real_dt else format_time(seconds) + ' into that video'}")
            print(f"    Logged at: {created_at}")
            report["last_known_sighting_previous"] = {
                "camera": camera,
                "video": video_source,
                "seconds_into_video": seconds,
                "real_datetime": real_dt,
                "similarity": similarity,
                "logged_at": created_at
            }
        else:
            print("  No sighting of this person has ever been recorded.")

    timestamp_tag = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(REPORTS_DIR, f"{args.person_id}_{timestamp_tag}.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("")
    print(f"Report saved: {report_path}")

    conn.close()


if __name__ == "__main__":
    main()
