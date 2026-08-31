import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "traffic_tracking.db"


def create_person_table():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS person_sightings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_reference TEXT NOT NULL,
            camera_id TEXT NOT NULL,
            city TEXT,
            road TEXT,
            video_source TEXT,
            first_seen TEXT,
            first_frame INTEGER,
            last_seen TEXT,
            last_frame INTEGER,
            best_similarity REAL,
            yolo_confidence REAL,
            matching_frames INTEGER
        )
    """)

    conn.commit()
    conn.close()

    print("Person sightings database ready.")


if __name__ == "__main__":
    print("=" * 70)
    print("SMART TRAFFIC MONITORING SYSTEM")
    print("PERSON DATABASE INITIALIZATION")
    print("=" * 70)

    create_person_table()

    print()
    print("Database:")
    print(DB_PATH)
    print()
    print("Person sightings table created successfully.")