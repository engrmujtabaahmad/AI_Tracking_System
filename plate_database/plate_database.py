import csv
import os

# ==========================================
# Database File
# ==========================================

DATABASE_FILE = "plate_database/plate_records.csv"


# ==========================================
# Create Database
# ==========================================

def create_database():

    os.makedirs("plate_database", exist_ok=True)

    if not os.path.exists(DATABASE_FILE):

        with open(
            DATABASE_FILE,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                "plate_number",
                "camera_id",
                "city",
                "road",
                "timestamp",
                "frame"
            ])

        print("Plate database created successfully.")

    else:

        print("Plate database already exists.")


# ==========================================
# Add Plate Record
# ==========================================

def add_record(
    plate_number,
    camera_id,
    city,
    road,
    timestamp,
    frame
):

    with open(
        DATABASE_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            plate_number,
            camera_id,
            city,
            road,
            timestamp,
            frame
        ])

    print("Record added successfully.")


# ==========================================
# Search Plate
# ==========================================

def search_plate(plate_number):

    if not os.path.exists(DATABASE_FILE):

        print("Database does not exist.")

        return []

    results = []

    with open(
        DATABASE_FILE,
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            if row["plate_number"].upper() == plate_number.upper():

                results.append(row)

    return results


# ==========================================
# Main Test
# ==========================================

if __name__ == "__main__":

    print("=" * 50)
    print("PLATE DATABASE")
    print("=" * 50)

    create_database()

    # --------------------------------------
    # Add sample records
    # --------------------------------------

    add_record(
        "NLTU389",
        "Camera_01",
        "Lahore",
        "Canal Road",
        "10:15:20",
        120
    )

    add_record(
        "NLTU389",
        "Camera_02",
        "Lahore",
        "Mall Road",
        "10:21:35",
        245
    )

    add_record(
        "NLTU389",
        "Camera_03",
        "Lahore",
        "Jail Road",
        "10:31:10",
        510
    )

    # --------------------------------------
    # Search
    # --------------------------------------

    plate = "NLTU389"

    results = search_plate(plate)

    print("\n" + "=" * 50)
    print("SEARCH RESULTS")
    print("=" * 50)

    if not results:

        print("Plate not found.")

    else:

        for result in results:

            print(
                f"Camera    : {result['camera_id']}"
            )

            print(
                f"City      : {result['city']}"
            )

            print(
                f"Road      : {result['road']}"
            )

            print(
                f"Timestamp : {result['timestamp']}"
            )

            print(
                f"Frame     : {result['frame']}"
            )

            print("-" * 50)

        # ----------------------------------
        # Last Seen
        # ----------------------------------

        last_seen = results[-1]

        print("\n" + "=" * 50)
        print("LAST SEEN LOCATION")
        print("=" * 50)

        print(
            f"Plate     : {last_seen['plate_number']}"
        )

        print(
            f"Camera    : {last_seen['camera_id']}"
        )

        print(
            f"City      : {last_seen['city']}"
        )

        print(
            f"Road      : {last_seen['road']}"
        )

        print(
            f"Time      : {last_seen['timestamp']}"
        )

        print(
            f"Frame     : {last_seen['frame']}"
        )

        print("=" * 50)