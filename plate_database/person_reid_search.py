import sqlite3
import os

# ============================================================
# SMART TRAFFIC MONITORING SYSTEM
# PERSON RE-ID DATABASE SEARCH
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "plate_database", "traffic_tracking.db")


def get_connection():
    return sqlite3.connect(DB_PATH)


def search_person(reference_image):
    conn = get_connection()
    conn.row_factory = sqlite3.Row

    cursor = conn.cursor()

    query = """
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
            matching_frames
        FROM person_reid_sightings
        WHERE reference_image = ?
        ORDER BY best_similarity DESC
    """

    cursor.execute(query, (reference_image,))
    results = cursor.fetchall()

    conn.close()

    return results


def print_result(reference_image, results):

    print("=" * 70)
    print("SMART TRAFFIC MONITORING SYSTEM")
    print("PERSON IMAGE SEARCH")
    print("=" * 70)

    print()
    print(f"Reference Image : {reference_image}")

    if not results:
        print()
        print("STATUS: NOT FOUND")
        print()
        print("The person was not found in any processed camera.")
        print("=" * 70)
        return

    print()
    print("STATUS: FOUND")
    print()
    print(f"Total Camera Sightings : {len(results)}")

    print()
    print("=" * 70)
    print("PERSON SIGHTINGS")
    print("=" * 70)

    for index, row in enumerate(results, start=1):

        print()
        print(f"Sighting #{index}")
        print("-" * 70)

        print(f"Camera              : {row['camera_id']}")
        print(f"City                : {row['city']}")
        print(f"Road                : {row['road']}")
        print(f"Video               : {row['video_source']}")

        print()
        print(f"First Seen          : {row['first_seen_timestamp']}")
        print(f"First Frame         : {row['first_seen_frame']}")

        print(f"Last Seen           : {row['last_seen_timestamp']}")
        print(f"Last Frame          : {row['last_seen_frame']}")

        print()
        print(f"Best Similarity     : {row['best_similarity']:.3f}")
        print(f"YOLO Confidence     : {row['best_yolo_confidence']:.3f}")
        print(f"Matching Frames     : {row['matching_frames']}")

    # --------------------------------------------------------
    # BEST / LAST KNOWN LOCATION
    # --------------------------------------------------------

    best = results[0]

    print()
    print("=" * 70)
    print("BEST MATCH")
    print("=" * 70)

    print(f"Camera              : {best['camera_id']}")
    print(f"City                : {best['city']}")
    print(f"Road                : {best['road']}")
    print(f"Video               : {best['video_source']}")
    print(f"First Seen          : {best['first_seen_timestamp']}")
    print(f"Last Seen           : {best['last_seen_timestamp']}")
    print(f"Best Similarity     : {best['best_similarity']:.3f}")

    print()
    print("=" * 70)
    print("PERSON SEARCH COMPLETED")
    print("=" * 70)


def main():

    print("=" * 70)
    print("SMART TRAFFIC MONITORING SYSTEM")
    print("PERSON IMAGE DATABASE SEARCH")
    print("=" * 70)

    print()
    reference_image = input(
        "Enter reference image name (example: person.jpg): "
    ).strip()

    if not reference_image:
        print()
        print("ERROR: No image name entered.")
        return

    results = search_person(reference_image)

    print_result(reference_image, results)


if __name__ == "__main__":
    main()