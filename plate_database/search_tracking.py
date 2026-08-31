import sqlite3
import os
import re
from datetime import datetime


# ============================================================
# CONFIGURATION
# ============================================================

DATABASE_PATH = "plate_database/traffic_tracking.db"


# ============================================================
# NORMALIZE PLATE NUMBER
# ============================================================

def normalize_plate(plate):
    """
    Converts different plate formats into one standard format.

    Examples:
        NI2022
        NI 2022
        ni-2022
        ni2022

    All become:
        NI2022
    """

    if not plate:
        return ""

    plate = plate.upper()

    # Remove spaces, hyphens and other special characters
    plate = re.sub(r"[^A-Z0-9]", "", plate)

    return plate


# ============================================================
# SEARCH DATABASE
# ============================================================

def search_plate(plate_number):

    plate_number = normalize_plate(plate_number)

    connection = sqlite3.connect(DATABASE_PATH)

    cursor = connection.cursor()

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
        WHERE UPPER(plate_number) = ?
        ORDER BY timestamp ASC
    """, (plate_number,))

    records = cursor.fetchall()

    connection.close()

    return records


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(plate_number, records):

    print()
    print("=" * 60)
    print("SMART TRAFFIC MONITORING SYSTEM")
    print("VEHICLE TRACKING SEARCH")
    print("=" * 60)

    print()

    print(f"Plate Number : {plate_number}")

    if not records:

        print("Status       : NOT FOUND")
        print()
        print("This vehicle was not found in the database.")

        print("=" * 60)

        return

    print("Status       : FOUND")
    print(f"Occurrences  : {len(records)}")

    print()
    print("-" * 60)
    print("ALL DETECTIONS")
    print("-" * 60)

    for index, record in enumerate(records, start=1):

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
        print(f"Detection #{index}")

        print(f"Camera       : {camera}")
        print(f"City         : {city}")
        print(f"Road         : {road}")
        print(f"Timestamp    : {timestamp}")
        print(f"Frame        : {frame}")

        if confidence is not None:
            print(f"Confidence   : {confidence:.2f}")

        if video:
            print(f"Video        : {video}")

    # ========================================================
    # LAST SEEN LOCATION
    # ========================================================

    last_record = records[-1]

    (
        last_plate,
        last_camera,
        last_city,
        last_road,
        last_timestamp,
        last_frame,
        last_confidence,
        last_video
    ) = last_record

    print()
    print("=" * 60)
    print("LAST SEEN LOCATION")
    print("=" * 60)

    print(f"Plate Number : {last_plate}")
    print(f"Camera       : {last_camera}")
    print(f"City         : {last_city}")
    print(f"Road         : {last_road}")
    print(f"Timestamp    : {last_timestamp}")
    print(f"Frame        : {last_frame}")

    if last_confidence is not None:
        print(f"Confidence   : {last_confidence:.2f}")

    if last_video:
        print(f"Video        : {last_video}")

    print("=" * 60)


# ============================================================
# MAIN PROGRAM
# ============================================================

def main():

    print("=" * 60)
    print("SMART TRAFFIC MONITORING SYSTEM")
    print("LICENSE PLATE / VEHICLE SEARCH")
    print("=" * 60)

    # Check database
    if not os.path.exists(DATABASE_PATH):

        print()
        print("ERROR: Database does not exist.")
        print()
        print("Please run:")
        print("python plate_database/tracking_database.py")

        return

    print()

    user_input = input("Enter license plate number: ")

    plate_number = normalize_plate(user_input)

    if not plate_number:

        print()
        print("ERROR: Invalid plate number.")

        return

    print()
    print("Searching database...")

    records = search_plate(plate_number)

    display_results(plate_number, records)


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":
    main()