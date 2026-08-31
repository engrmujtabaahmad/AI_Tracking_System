from services.hospital_service import (
    load_hospitals,
    find_hospital_by_name
)

from services.route_service import get_routes


# --------------------------------------------------
# USER STARTING LOCATION
# --------------------------------------------------

START_LAT = 31.5204
START_LON = 74.3587


# --------------------------------------------------
# HOSPITAL TO SEARCH
# --------------------------------------------------

HOSPITAL_NAME = "PKLI"


print("=" * 70)
print("LAHORE HOSPITAL ROUTING SYSTEM")
print("=" * 70)


# --------------------------------------------------
# LOAD HOSPITALS
# --------------------------------------------------

hospitals = load_hospitals()

print(f"\nTotal hospital records: {len(hospitals)}")


# --------------------------------------------------
# SEARCH HOSPITAL
# --------------------------------------------------

matches = find_hospital_by_name(HOSPITAL_NAME)


if not matches:

    print(
        f"\nNo hospital found matching: {HOSPITAL_NAME}"
    )

    exit()


print(
    f"\nHospitals matching '{HOSPITAL_NAME}': "
    f"{len(matches)}"
)


for i, hospital in enumerate(matches):

    print("\n" + "-" * 70)

    print(f"Hospital {i + 1}")

    print(
        "Name      :",
        hospital.get("name", "Unknown")
    )

    print(
        "Latitude  :",
        hospital.get("latitude")
    )

    print(
        "Longitude :",
        hospital.get("longitude")
    )

    print(
        "Address   :",
        hospital.get("address", "Not available")
    )


# --------------------------------------------------
# SELECT FIRST MATCH
# --------------------------------------------------

hospital = matches[0]


destination_lat = hospital["latitude"]
destination_lon = hospital["longitude"]


print("\n" + "=" * 70)

print("ROUTE CALCULATION")

print("=" * 70)

print("\nStarting location:")

print("Latitude :", START_LAT)
print("Longitude:", START_LON)


print("\nDestination hospital:")

print("Name:", hospital["name"])

print("Latitude :", destination_lat)
print("Longitude:", destination_lon)


# --------------------------------------------------
# GET ROUTES
# --------------------------------------------------

print("\nCalculating alternative routes...")


routes = get_routes(
    START_LAT,
    START_LON,
    destination_lat,
    destination_lon
)


# --------------------------------------------------
# DISPLAY ROUTES
# --------------------------------------------------

if not routes:

    print("\nERROR: No routes were returned.")

else:

    print(
        f"\nSUCCESS: {len(routes)} route(s) received."
    )

    for route in routes:

        print("\n" + "-" * 70)

        print(
            f"ROUTE {route['route_number']}"
        )

        print(
            f"Distance : "
            f"{route['distance_km']} km"
        )

        print(
            f"Time     : "
            f"{route['duration_minutes']} minutes"
        )