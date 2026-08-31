from services.route_service import get_routes


# Example starting location in Lahore
start_lat = 31.5204
start_lon = 74.3587


# Example destination
# PKLI coordinates from our hospital dataset
destination_lat = 31.4557
destination_lon = 74.4621886


print("=" * 60)
print("LAHORE ROUTING TEST")
print("=" * 60)

print("\nStarting location:")
print(f"Latitude : {start_lat}")
print(f"Longitude: {start_lon}")

print("\nDestination:")
print(f"Latitude : {destination_lat}")
print(f"Longitude: {destination_lon}")

print("\nCalculating routes...\n")


routes = get_routes(
    start_lat,
    start_lon,
    destination_lat,
    destination_lon
)


if not routes:

    print("ERROR: No routes found.")

else:

    print(f"SUCCESS: {len(routes)} route(s) received.\n")

    for route in routes:

        print("-" * 60)

        print(
            f"Route {route['route_number']}"
        )

        print(
            f"Distance: "
            f"{route['distance_km']} km"
        )

        print(
            f"Estimated time: "
            f"{route['duration_minutes']} minutes"
        )

    print("-" * 60)