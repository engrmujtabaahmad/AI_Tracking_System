import sqlite3
import os
from datetime import datetime


# ============================================================
# CONFIGURATION
# ============================================================

DATABASE_PATH = "plate_database/traffic_tracking.db"


# ============================================================
# CREATE DATABASE DIRECTORY
# ============================================================

os.makedirs("plate_database", exist_ok=True)


# ============================================================
# CONNECT TO DATABASE
# ============================================================

connection = sqlite3.connect(DATABASE_PATH)

cursor = connection.cursor()


# ============================================================
# CREATE TABLE
# ============================================================

cursor.execute("""
CREATE TABLE IF NOT EXISTS vehicle_detections (

    id INTEGER PRIMARY KEY AUTOINCREMENT,

    plate_number TEXT NOT NULL,

    camera_id TEXT NOT NULL,

    city TEXT NOT NULL,

    road TEXT NOT NULL,

    timestamp TEXT NOT NULL,

    frame INTEGER,

    confidence REAL,

    video_source TEXT,

    created_at TEXT DEFAULT CURRENT_TIMESTAMP

)
""")


connection.commit()


print("=" * 60)
print("SMART TRAFFIC TRACKING DATABASE")
print("=" * 60)

print("Database created successfully.")

print()
print(f"Database location:")
print(DATABASE_PATH)


# ============================================================
# FUNCTION: ADD DETECTION
# ============================================================

def add_detection(
    plate_number,
    camera_id,
    city,
    road,
    timestamp,
    frame=None,
    confidence=None,
    video_source=None
):

    cursor.execute("""
    INSERT INTO vehicle_detections
    (
        plate_number,
        camera_id,
        city,
        road,
        timestamp,
        frame,
        confidence,
        video_source
    )

    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (

        plate_number.upper().replace(" ", "").replace("-", ""),

        camera_id,

        city,

        road,

        timestamp,

        frame,

        confidence,

        video_source

    ))

    connection.commit()


# ============================================================
# ADD TEST DATA
# ============================================================

print()
print("=" * 60)
print("ADDING TEST CAMERA RECORDS")
print("=" * 60)


add_detection(
    plate_number="NI2022",
    camera_id="Camera_01",
    city="Lahore",
    road="Canal Road",
    timestamp="10:15:20",
    frame=120,
    confidence=0.92,
    video_source="Traffic1.mp4"
)


add_detection(
    plate_number="NI2022",
    camera_id="Camera_02",
    city="Lahore",
    road="Mall Road",
    timestamp="10:21:35",
    frame=245,
    confidence=0.89,
    video_source="Traffic2.mp4"
)


add_detection(
    plate_number="NI2022",
    camera_id="Camera_03",
    city="Lahore",
    road="Jail Road",
    timestamp="10:31:10",
    frame=510,
    confidence=0.94,
    video_source="Traffic3.mp4"
)


add_detection(
    plate_number="ABC999",
    camera_id="Camera_01",
    city="Lahore",
    road="Canal Road",
    timestamp="11:05:20",
    frame=800,
    confidence=0.91,
    video_source="Traffic1.mp4"
)


print("Test records added successfully.")


# ============================================================
# DISPLAY ALL RECORDS
# ============================================================

print()
print("=" * 60)
print("ALL DATABASE RECORDS")
print("=" * 60)


cursor.execute("""
SELECT
    plate_number,
    camera_id,
    city,
    road,
    timestamp,
    frame,
    confidence,
    video_source

FROM vehicle_detections

ORDER BY timestamp
""")


records = cursor.fetchall()


for record in records:

    (
        plate,
        camera,
        city,
        road,
        timestamp,
        frame,
        confidence,
        video
    ) = record


    print()
    print(f"Plate       : {plate}")
    print(f"Camera      : {camera}")
    print(f"City        : {city}")
    print(f"Road        : {road}")
    print(f"Timestamp   : {timestamp}")
    print(f"Frame       : {frame}")
    print(f"Confidence  : {confidence}")
    print(f"Video       : {video}")


# ============================================================
# CLOSE DATABASE
# ============================================================

connection.close()


print()
print("=" * 60)
print("DATABASE SETUP COMPLETED")
print("=" * 60)