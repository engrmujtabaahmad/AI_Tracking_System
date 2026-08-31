import sqlite3
from pathlib import Path
from datetime import datetime


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATABASE = BASE_DIR / "traffic_tracking.db"


# ============================================================
# DATABASE CONNECTION
# ============================================================

def connect_database():
    if not DATABASE.exists():
        print("\nERROR: Database does not exist.")
        print(f"Expected location: {DATABASE}")
        return None

    return sqlite3.connect(DATABASE)


# ============================================================
# NORMALIZE PLATE NUMBER
# ============================================================

def normalize_plate(plate):
    """
    Normalize user input.

    Example:
        ni-2022
        NI 2022
        NI2022

    All become:
        NI2022
    """

    if not plate:
        return ""

    plate = plate.upper()

    # Remove spaces and common separators
    plate = plate.replace(" ", "")
    plate = plate.replace("-", "")
    plate = plate.replace("_", "")

    return plate


# ============================================================
# GET VEHICLE RECORDS
# ============================================================

def get_vehicle_records(plate_number):

    conn = connect_database()

    if conn is None:
        return []

    cursor = conn.cursor()

    query = """
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
        ORDER BY rowid ASC
    """

    cursor.execute(query)

    rows = cursor.fetchall()

    conn.close()

    normalized_search = normalize_plate(plate_number)

    results = []

    for row in rows:

        database_plate = normalize_plate(row[0])

        if database_plate == normalized_search:
            results.append(row)

    return results


# ============================================================
# CALCULATE AVERAGE CONFIDENCE
# ============================================================

def calculate_average_confidence(records):

    if not records:
        return 0.0

    total = sum(float(record[6]) for record in records)

    return total / len(records)


# ============================================================
# DISPLAY REPORT
# ============================================================

def display_report(plate_number, records):

    print("\n")
    print("=" * 70)
    print("SMART TRAFFIC MONITORING SYSTEM")
    print("VEHICLE TRACKING REPORT")
    print("=" * 70)

    if not records:

        print()
        print(f"Plate Number : {normalize_plate(plate_number)}")
        print("Status       : NOT FOUND")

        print()
        print("No detection records were found for this vehicle.")

        print("=" * 70)

        return

    normalized_plate = normalize_plate(plate_number)

    average_confidence = calculate_average_confidence(records)

    print()
    print(f"Plate Number     : {normalized_plate}")
    print("Status           : FOUND")
    print(f"Total Detections : {len(records)}")
    print(
        f"Average Confidence: {average_confidence:.2f}"
    )

    print()
    print("-" * 70)
    print("VEHICLE JOURNEY")
    print("-" * 70)

    for index, record in enumerate(records, start=1):

        plate = record[0]
        camera = record[1]
        city = record[2]
        road = record[3]
        timestamp = record[4]
        frame = record[5]
        confidence = record[6]
        video = record[7]

        print()
        print(f"Detection #{index}")
        print(f"Plate       : {plate}")
        print(f"Camera      : {camera}")
        print(f"City        : {city}")
        print(f"Road        : {road}")
        print(f"Timestamp   : {timestamp}")
        print(f"Frame       : {frame}")
        print(f"Confidence  : {confidence:.2f}")
        print(f"Video       : {video}")

    # --------------------------------------------------------
    # FIRST DETECTION
    # --------------------------------------------------------

    first = records[0]

    print()
    print("=" * 70)
    print("FIRST SEEN")
    print("=" * 70)

    print(f"Camera    : {first[1]}")
    print(f"City      : {first[2]}")
    print(f"Road      : {first[3]}")
    print(f"Timestamp : {first[4]}")
    print(f"Frame     : {first[5]}")
    print(f"Video     : {first[7]}")

    # --------------------------------------------------------
    # LAST DETECTION
    # --------------------------------------------------------

    last = records[-1]

    print()
    print("=" * 70)
    print("LAST SEEN")
    print("=" * 70)

    print(f"Camera     : {last[1]}")
    print(f"City       : {last[2]}")
    print(f"Road       : {last[3]}")
    print(f"Timestamp  : {last[4]}")
    print(f"Frame      : {last[5]}")
    print(f"Confidence : {last[6]:.2f}")
    print(f"Video      : {last[7]}")

    # --------------------------------------------------------
    # CAMERA HISTORY
    # --------------------------------------------------------

    cameras = []

    for record in records:

        camera = record[1]

        if camera not in cameras:
            cameras.append(camera)

    print()
    print("=" * 70)
    print("CAMERA HISTORY")
    print("=" * 70)

    for camera in cameras:

        camera_records = [
            record
            for record in records
            if record[1] == camera
        ]

        print(
            f"{camera} : {len(camera_records)} detection(s)"
        )

    # --------------------------------------------------------
    # ROAD HISTORY
    # --------------------------------------------------------

    roads = []

    for record in records:

        road = record[3]

        if road not in roads:
            roads.append(road)

    print()
    print("=" * 70)
    print("ROAD HISTORY")
    print("=" * 70)

    for road in roads:

        road_records = [
            record
            for record in records
            if record[3] == road
        ]

        print(
            f"{road} : {len(road_records)} detection(s)"
        )

    print()
    print("=" * 70)
    print("REPORT COMPLETED")
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("SMART TRAFFIC MONITORING SYSTEM")
    print("VEHICLE TRACKING REPORT GENERATOR")
    print("=" * 70)

    print()

    plate = input(
        "Enter license plate number: "
    ).strip()

    if not plate:

        print()
        print("ERROR: Plate number cannot be empty.")
        return

    print()
    print("Searching database...")

    records = get_vehicle_records(plate)

    display_report(plate, records)


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":
    main()