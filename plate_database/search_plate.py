import csv
import os

# ==========================================
# Database
# ==========================================

DATABASE_FILE = "plate_database/plate_records.csv"


# ==========================================
# Search Plate
# ==========================================

def search_plate(plate_number):

    if not os.path.exists(DATABASE_FILE):
        print("Database not found.")
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

            if row["plate_number"].strip().upper() == plate_number.strip().upper():
                results.append(row)

    return results


# ==========================================
# Display Results
# ==========================================

def display_results(plate_number, results):

    print("\n" + "=" * 60)
    print("LICENSE PLATE SEARCH")
    print("=" * 60)

    print(f"Plate Number : {plate_number}")

    if not results:

        print("\nStatus       : NOT FOUND")
        print("\nThis plate was not found in the database.")

        print("=" * 60)
        return

    print(f"Status       : FOUND")
    print(f"Occurrences  : {len(results)}")

    print("\n" + "-" * 60)
    print("ALL DETECTIONS")
    print("-" * 60)

    for i, result in enumerate(results, start=1):

        print(f"\nDetection #{i}")

        print(f"Camera       : {result['camera_id']}")
        print(f"City         : {result['city']}")
        print(f"Road         : {result['road']}")
        print(f"Timestamp    : {result['timestamp']}")
        print(f"Frame        : {result['frame']}")

    # ======================================
    # Last Seen
    # ======================================

    last_seen = results[-1]

    print("\n" + "=" * 60)
    print("LAST SEEN LOCATION")
    print("=" * 60)

    print(f"Plate Number : {last_seen['plate_number']}")
    print(f"Camera       : {last_seen['camera_id']}")
    print(f"City         : {last_seen['city']}")
    print(f"Road         : {last_seen['road']}")
    print(f"Timestamp    : {last_seen['timestamp']}")
    print(f"Frame        : {last_seen['frame']}")

    print("=" * 60)


# ==========================================
# Main Program
# ==========================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("SMART TRAFFIC MONITORING SYSTEM")
    print("LICENSE PLATE SEARCH")
    print("=" * 60)

    plate_number = input(
        "\nEnter license plate number: "
    ).strip()

    if not plate_number:

        print("Please enter a license plate number.")

    else:

        results = search_plate(plate_number)

        display_results(
            plate_number,
            results
        )