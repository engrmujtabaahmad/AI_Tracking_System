from services.hospital_service import (
    find_nearest_hospitals
)

from services.route_service import (
    calculate_routes,
    get_fastest_route
)

from utils.map_utils import (
    create_route_map
)


# ============================================================
# ACCIDENT LOCATION
# ============================================================

ACCIDENT_LATITUDE = 31.5204
ACCIDENT_LONGITUDE = 74.3587


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("INTEGRATED ACCIDENT RESPONSE MAP")
print("=" * 60)


# ============================================================
# FIND NEAREST HOSPITALS
# ============================================================

print("\nSearching for nearest hospitals...")

hospitals = find_nearest_hospitals(
    ACCIDENT_LATITUDE,
    ACCIDENT_LONGITUDE,
    limit=5
)


if not hospitals:

    print("\nNo hospitals found.")

    exit()


# ============================================================
# SELECT NEAREST HOSPITAL
# ============================================================

nearest_hospital = hospitals[0]

hospital_latitude = nearest_hospital["latitude"]
hospital_longitude = nearest_hospital["longitude"]


print("\nNearest Hospital:")
print(f"Name     : {nearest_hospital['name']}")
print(f"Distance : {nearest_hospital['distance_km']} km")
print(f"Address  : {nearest_hospital.get('address')}")


# ============================================================
# CALCULATE ROUTE
# ============================================================

print("\nCalculating route...")

routes = calculate_routes(
    ACCIDENT_LATITUDE,
    ACCIDENT_LONGITUDE,
    hospital_latitude,
    hospital_longitude
)


if not routes:

    print("\nNo route found.")

    exit()


print(f"{len(routes)} route(s) received.")


# ============================================================
# SELECT FASTEST ROUTE
# ============================================================

fastest_route = get_fastest_route(routes)


print("\nRecommended Route:")
print(
    f"Distance : "
    f"{fastest_route['distance_km']} km"
)

print(
    f"Time     : "
    f"{fastest_route['duration_minutes']} minutes"
)


# ============================================================
# CREATE MAP
# ============================================================

print("\nCreating integrated map...")


route_map = create_route_map(
    ACCIDENT_LATITUDE,
    ACCIDENT_LONGITUDE,
    nearest_hospital,
    routes
)


# ============================================================
# SAVE MAP
# ============================================================

output_file = (
    "problem2_integrated_accident_map.html"
)


route_map.save(output_file)


print("\nMap created successfully.")

print(
    f"Saved as: {output_file}"
)


# ============================================================
# FINAL INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("FINAL RESULT")
print("=" * 60)

print(
    f"\nAccident Location:"
    f"\nLatitude  : {ACCIDENT_LATITUDE}"
    f"\nLongitude : {ACCIDENT_LONGITUDE}"
)

print(
    f"\nNearest Hospital:"
    f"\n{nearest_hospital['name']}"
)

print(
    f"\nHospital Distance:"
    f"\n{nearest_hospital['distance_km']} km"
)

print(
    f"\nRoad Distance:"
    f"\n{fastest_route['distance_km']} km"
)

print(
    f"\nEstimated Travel Time:"
    f"\n{fastest_route['duration_minutes']} minutes"
)

print("\nIntegrated map is ready.")

print("=" * 60)