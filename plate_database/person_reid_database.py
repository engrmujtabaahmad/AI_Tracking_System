import sqlite3
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "traffic_tracking.db"


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_connection():
    return sqlite3.connect(DB_PATH)


# ============================================================
# CREATE PERSON RE-ID TABLE
# ============================================================

def create_person_reid_table():

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS person_reid_sightings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            reference_image TEXT NOT NULL,

            camera_id TEXT NOT NULL,
            city TEXT,
            road TEXT,

            video_source TEXT NOT NULL,

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
    conn.close()

    print("Person Re-ID database ready.")
    print()
    print(f"Database:")
    print(DB_PATH)
    print()
    print("Person Re-ID sightings table created successfully.")


# ============================================================
# SAVE PERSON SIGHTING
# ============================================================

def save_person_sighting(
    reference_image,
    camera_id,
    city,
    road,
    video_source,
    first_seen_timestamp,
    first_seen_frame,
    last_seen_timestamp,
    last_seen_frame,
    best_similarity,
    best_yolo_confidence,
    matching_frames
):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO person_reid_sightings (
            reference_image,
            camera_id,
            city,
            road,
            video_source,
            first_seen_timestamp,
            first_seen_frame,
            last_seen_timestamp,
            last_seen_frame,
            best_similarity,
            best_yolo_confidence,
            matching_frames
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        reference_image,
        camera_id,
        city,
        road,
        video_source,
        first_seen_timestamp,
        first_seen_frame,
        last_seen_timestamp,
        last_seen_frame,
        best_similarity,
        best_yolo_confidence,
        matching_frames
    ))

    conn.commit()
    conn.close()


# ============================================================
# GET ALL RE-ID SIGHTINGS
# ============================================================

def get_all_sightings():

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            reference_image,
            camera_id,
            city,
            road,
            video_source,
            first_seen_timestamp,
            first_seen_frame,
            last_seen_timestamp,
            last_seen_frame,
            best_similarity,
            best_yolo_confidence,
            matching_frames,
            created_at
        FROM person_reid_sightings
        ORDER BY created_at DESC
    """)

    results = cursor.fetchall()

    conn.close()

    return results


# ============================================================
# SEARCH BY REFERENCE IMAGE
# ============================================================

def search_by_reference_image(reference_image):

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            camera_id,
            city,
            road,
            video_source,
            first_seen_timestamp,
            first_seen_frame,
            last_seen_timestamp,
            last_seen_frame,
            best_similarity,
            best_yolo_confidence,
            matching_frames,
            created_at
        FROM person_reid_sightings
        WHERE reference_image = ?
        ORDER BY best_similarity DESC
    """, (reference_image,))

    results = cursor.fetchall()

    conn.close()

    return results


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("SMART TRAFFIC MONITORING SYSTEM")
    print("PERSON RE-ID DATABASE")
    print("=" * 70)
    print()

    create_person_reid_table()