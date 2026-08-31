import json
import math
from pathlib import Path


# ============================================================
# HOSPITAL DATA FILE
# ============================================================

HOSPITAL_FILE = (
    Path(__file__).resolve()
    .parents[2]
    / "data"
    / "problem2"
    / "lahore_hospitals.json"
)


# ============================================================
# LOAD HOSPITALS
# ============================================================

def load_hospitals():

    if not HOSPITAL_FILE.exists():

        raise FileNotFoundError(
            f"Hospital file not found:\n{HOSPITAL_FILE}"
        )


    with open(
        HOSPITAL_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        hospitals = json.load(file)


    return hospitals


# ============================================================
# HOSPITAL COUNT
# ============================================================

def get_hospital_count():

    hospitals = load_hospitals()

    return len(hospitals)


# ============================================================
# SEARCH HOSPITAL BY NAME
# ============================================================

def find_hospital_by_name(name):

    hospitals = load_hospitals()

    search_text = (
        name.lower().strip()
    )

    results = []


    for hospital in hospitals:

        hospital_name = hospital.get(
            "name",
            ""
        )


        if search_text in hospital_name.lower():

            results.append(
                hospital
            )


    return results


# ============================================================
# GET HOSPITAL BY INDEX
# ============================================================

def get_hospital_by_index(index):

    hospitals = load_hospitals()


    if (
        index < 0
        or index >= len(hospitals)
    ):

        return None


    return hospitals[index]


# ============================================================
# DISTANCE CALCULATION
# ============================================================

def calculate_distance(
    lat1,
    lon1,
    lat2,
    lon2
):

    earth_radius = 6371.0


    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)

    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)


    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1


    a = (
        math.sin(delta_lat / 2) ** 2
        +
        math.cos(lat1)
        *
        math.cos(lat2)
        *
        math.sin(delta_lon / 2) ** 2
    )


    c = 2 * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a)
    )


    return earth_radius * c


# ============================================================
# FIND NEAREST HOSPITALS
# ============================================================

def find_nearest_hospitals(
    latitude,
    longitude,
    limit=5
):

    hospitals = load_hospitals()

    hospitals_with_distance = []


    for hospital in hospitals:

        hospital_lat = hospital.get(
            "latitude"
        )

        hospital_lon = hospital.get(
            "longitude"
        )


        if (
            hospital_lat is None
            or hospital_lon is None
        ):

            continue


        try:

            hospital_lat = float(
                hospital_lat
            )

            hospital_lon = float(
                hospital_lon
            )

        except (
            TypeError,
            ValueError
        ):

            continue


        distance = calculate_distance(
            latitude,
            longitude,
            hospital_lat,
            hospital_lon
        )


        hospital_copy = hospital.copy()


        hospital_copy[
            "distance_km"
        ] = round(
            distance,
            2
        )


        hospitals_with_distance.append(
            hospital_copy
        )


    hospitals_with_distance.sort(
        key=lambda hospital:
        hospital["distance_km"]
    )


    return (
        hospitals_with_distance[
            :limit
        ]
    )