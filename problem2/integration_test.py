from services.hospital_service import find_nearest_hospitals
from services.route_service import (
    calculate_routes,
    get_fastest_route,
    get_shortest_route
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
print("ACCIDENT RESPONSE SYSTEM")
print("=" * 60)


# ============================================================
# ACCIDENT LOCATION
# ============================================================

print("\nAccident Location:")
print(f"Latitude  : {ACCIDENT_LATITUDE}")
print(f"Longitude : {ACCIDENT_LONGITUDE}")


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
    print("\nERROR: No hospitals found.")
    exit()


print("\nNearest Hospitals:")
print("-" * 60)

for i, hospital in enumerate(hospitals, start=1):

    print(f"\n#{i}")
    print(f"Name     : {hospital.get('name')}")
    print(f"Distance : {hospital.get('distance_km')} km")
    print(f"Address  : {hospital.get('address')}")


# ============================================================
# SELECT NEAREST HOSPITAL
# ============================================================

nearest_hospital = hospitals[0]

hospital_latitude = nearest_hospital["latitude"]
hospital_longitude = nearest_hospital["longitude"]


print("\n")
print("=" * 60)
print("NEAREST HOSPITAL")
print("=" * 60)

print(f"Hospital : {nearest_hospital['name']}")
print(f"Distance : {nearest_hospital['distance_km']} km")
print(f"Address  : {nearest_hospital.get('address')}")


# ============================================================
# CALCULATE ROUTES
# ============================================================

print("\nCalculating routes to nearest hospital...")

routes = calculate_routes(
    ACCIDENT_LATITUDE,
    ACCIDENT_LONGITUDE,
    hospital_latitude,
    hospital_longitude
)


if not routes:
    print("\nERROR: No route could be calculated.")
    exit()


print(f"\n{len(routes)} route(s) received.")


# ============================================================
# FASTEST ROUTE
# ============================================================

fastest = get_fastest_route(routes)


# ============================================================
# SHORTEST ROUTE
# ============================================================

shortest = get_shortest_route(routes)


# ============================================================
# ROUTE COMPARISON
# ============================================================

print("\n")
print("=" * 60)
print("ROUTE ANALYSIS")
print("=" * 60)

print("\nFastest Route:")
print(f"Distance : {fastest['distance_km']} km")
print(f"Time     : {fastest['duration_minutes']} minutes")

print("\nShortest Route:")
print(f"Distance : {shortest['distance_km']} km")
print(f"Time     : {shortest['duration_minutes']} minutes")


# ============================================================
# RECOMMENDED ROUTE
# ============================================================

recommended_route = fastest


print("\n")
print("=" * 60)
print("RECOMMENDED ROUTE")
print("=" * 60)

print(f"Distance : {recommended_route['distance_km']} km")
print(f"Time     : {recommended_route['duration_minutes']} minutes")


# ============================================================
# FINAL EMERGENCY RESPONSE
# ============================================================

print("\n")
print("=" * 60)
print("FINAL EMERGENCY RESPONSE")
print("=" * 60)

print("\nAccident detected at:")
print(
    f"{ACCIDENT_LATITUDE}, "
    f"{ACCIDENT_LONGITUDE}"
)

print("\nNearest hospital:")
print(nearest_hospital["name"])

print(
    f"Hospital distance: "
    f"{nearest_hospital['distance_km']} km"
)

print(
    f"Road distance: "
    f"{recommended_route['distance_km']} km"
)

print(
    f"Estimated travel time: "
    f"{recommended_route['duration_minutes']} minutes"
)

print("\nEmergency response required.")
print("\nRoute calculation completed successfully.")

print("=" * 60)