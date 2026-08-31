import csv
import json
import time
from pathlib import Path

import requests


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "data" / "problem2"

JSON_FILE = OUTPUT_DIR / "lahore_hospitals.json"
CSV_FILE = OUTPUT_DIR / "lahore_hospitals.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LAHORE BOUNDING BOX
# ============================================================

SOUTH = 31.25
WEST = 74.05
NORTH = 31.70
EAST = 74.60


# ============================================================
# OVERPASS SERVERS
# ============================================================

OVERPASS_SERVERS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
]


# ============================================================
# REQUEST SETTINGS
# ============================================================

HEADERS = {
    "User-Agent": "SmartTrafficMonitoringSystem/1.0"
}

REQUEST_TIMEOUT = 180


# ============================================================
# QUERY
# ============================================================
#
# We deliberately start with NODES only.
#
# This is much lighter than requesting nodes + ways +
# relations.
#
# Most hospitals have a hospital point/node in OSM.
# ============================================================

OVERPASS_QUERY = f"""
[out:json][timeout:120];

(
    node["amenity"="hospital"]({SOUTH},{WEST},{NORTH},{EAST});
    node["healthcare"="hospital"]({SOUTH},{WEST},{NORTH},{EAST});
);

out body;
"""


# ============================================================
# GET TAG
# ============================================================

def get_tag(tags, *keys):

    for key in keys:

        value = tags.get(key)

        if value:
            return str(value).strip()

    return ""


# ============================================================
# FETCH DATA
# ============================================================

def fetch_hospitals():

    print("=" * 70)
    print("LAHORE HOSPITAL DATA COLLECTION")
    print("=" * 70)

    print()
    print("Source       : OpenStreetMap")
    print("Service      : Overpass API")
    print("Area         : Lahore")
    print()
    print("Searching for hospitals...")
    print()

    last_error = None

    for server in OVERPASS_SERVERS:

        print("-" * 70)
        print("Trying server:")
        print(server)
        print()

        try:

            response = requests.post(
                server,
                data=OVERPASS_QUERY,
                headers=HEADERS,
                timeout=REQUEST_TIMEOUT,
            )

            response.raise_for_status()

            data = response.json()

            elements = data.get("elements", [])

            print(
                f"SUCCESS: {len(elements)} "
                f"hospital records received."
            )

            return elements

        except requests.exceptions.Timeout as error:

            last_error = error

            print("Request timed out.")
            print("Trying the next server...")
            print()

        except requests.exceptions.RequestException as error:

            last_error = error

            print("Server request failed:")
            print(error)
            print("Trying the next server...")
            print()

        except ValueError as error:

            last_error = error

            print("Server returned invalid JSON:")
            print(error)
            print()

        time.sleep(2)

    raise RuntimeError(
        "All Overpass servers failed. "
        f"Last error: {last_error}"
    )


# ============================================================
# PROCESS HOSPITALS
# ============================================================

def process_hospitals(elements):

    hospitals = []

    for element in elements:

        latitude = element.get("lat")
        longitude = element.get("lon")

        if latitude is None or longitude is None:
            continue

        tags = element.get("tags", {})

        name = get_tag(
            tags,
            "name",
            "name:en",
            "official_name"
        )

        if not name:
            name = "Unnamed Hospital"

        address = get_tag(
            tags,
            "addr:full"
        )

        street = get_tag(
            tags,
            "addr:street"
        )

        city = get_tag(
            tags,
            "addr:city"
        )

        postcode = get_tag(
            tags,
            "addr:postcode"
        )

        phone = get_tag(
            tags,
            "phone",
            "contact:phone"
        )

        website = get_tag(
            tags,
            "website",
            "contact:website"
        )

        operator = get_tag(
            tags,
            "operator"
        )

        emergency = get_tag(
            tags,
            "emergency"
        )

        hospital_type = get_tag(
            tags,
            "healthcare",
            "amenity"
        )

        if not address:

            parts = []

            if street:
                parts.append(street)

            if city:
                parts.append(city)

            if postcode:
                parts.append(postcode)

            address = ", ".join(parts)

        hospital = {

            "hospital_id": f"osm_node_{element['id']}",

            "name": name,

            "address": address,

            "city": city if city else "Lahore",

            "postcode": postcode,

            "phone": phone,

            "website": website,

            "operator": operator,

            "emergency": emergency,

            "type": hospital_type,

            "latitude": float(latitude),

            "longitude": float(longitude),

            "source": "OpenStreetMap",

        }

        hospitals.append(hospital)

    return hospitals


# ============================================================
# REMOVE DUPLICATES
# ============================================================

def remove_duplicates(hospitals):

    unique = {}

    for hospital in hospitals:

        key = (

            hospital["name"].lower().strip(),

            round(hospital["latitude"], 6),

            round(hospital["longitude"], 6),

        )

        if key not in unique:

            unique[key] = hospital

    return list(unique.values())


# ============================================================
# SORT
# ============================================================

def sort_hospitals(hospitals):

    hospitals.sort(
        key=lambda item: item["name"].lower()
    )

    return hospitals


# ============================================================
# SAVE JSON
# ============================================================

def save_json(hospitals):

    with open(
        JSON_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            hospitals,
            file,
            indent=4,
            ensure_ascii=False
        )

    print()
    print("JSON file saved:")
    print(JSON_FILE)


# ============================================================
# SAVE CSV
# ============================================================

def save_csv(hospitals):

    fields = [

        "hospital_id",
        "name",
        "address",
        "city",
        "postcode",
        "phone",
        "website",
        "operator",
        "emergency",
        "type",
        "latitude",
        "longitude",
        "source",

    ]

    with open(
        CSV_FILE,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields
        )

        writer.writeheader()

        writer.writerows(hospitals)

    print()
    print("CSV file saved:")
    print(CSV_FILE)


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_hospitals(hospitals):

    print()
    print("=" * 70)
    print("LAHORE HOSPITALS FOUND")
    print("=" * 70)

    print()
    print(f"Total hospitals: {len(hospitals)}")
    print()

    for index, hospital in enumerate(
        hospitals,
        start=1
    ):

        print(
            f"{index:03d}. "
            f"{hospital['name']}"
        )

        print(
            f"     Latitude  : "
            f"{hospital['latitude']}"
        )

        print(
            f"     Longitude : "
            f"{hospital['longitude']}"
        )

        if hospital["address"]:

            print(
                f"     Address   : "
                f"{hospital['address']}"
            )

        if hospital["phone"]:

            print(
                f"     Phone     : "
                f"{hospital['phone']}"
            )

        print()


# ============================================================
# MAIN
# ============================================================

def main():

    try:

        elements = fetch_hospitals()

        hospitals = process_hospitals(elements)

        hospitals = remove_duplicates(hospitals)

        hospitals = sort_hospitals(hospitals)

        display_hospitals(hospitals)

        save_json(hospitals)

        save_csv(hospitals)

        print()
        print("=" * 70)
        print("HOSPITAL DATA COLLECTION COMPLETED")
        print("=" * 70)

        print()
        print(
            "Hospital data is ready for the "
            "Problem 2 routing system."
        )

        print()

    except KeyboardInterrupt:

        print()
        print("Process stopped by user.")

    except Exception as error:

        print()
        print("=" * 70)
        print("ERROR")
        print("=" * 70)

        print()
        print(error)

        print()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()