from services.hospital_service import (
    get_hospital_count,
    find_nearest_hospitals
)


# ============================================================
# TEST LOCATION
# ============================================================

# Example accident/start location in Lahore
ACCIDENT_LAT = 31.5204
ACCIDENT_LON = 74.3587


print("=" * 70)
print("LAHORE NEAREST HOSPITAL SEARCH")
print("=" * 70)


# ============================================================
# HOSPITAL COUNT
# ============================================================

total = get_hospital_count()

print(
    f"\nTotal hospital records: {total}"
)


# ============================================================
# FIND NEAREST HOSPITALS
# ============================================================

print("\nSearching for nearest hospitals...")

hospitals = find_nearest_hospitals(
    ACCIDENT_LAT,
    ACCIDENT_LON,
    limit=5
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print(
    f"\nNearest {len(hospitals)} hospitals:"
)


for index, hospital in enumerate(
    hospitals,
    start=1
):

    print("\n" + "-" * 70)

    print(
        f"#{index}"
    )

    print(
        "Name      :",
        hospital.get(
            "name",
            "Unknown"
        )
    )

    print(
        "Distance  :",
        hospital.get(
            "distance_km"
        ),
        "km"
    )

    print(
        "Latitude  :",
        hospital.get(
            "latitude"
        )
    )

    print(
        "Longitude :",
        hospital.get(
            "longitude"
        )
    )

    print(
        "Address   :",
        hospital.get(
            "address",
            "Not available"
        )
    )


print("\n" + "=" * 70)
print("SEARCH COMPLETE")
print("=" * 70)