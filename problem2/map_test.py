from services.hospital_service import (
    find_hospital_by_name
)

from services.route_service import (
    get_routes
)

from utils.map_utils import (
    create_route_map
)


# --------------------------------------------------
# START LOCATION
# --------------------------------------------------

START_LAT = 31.5204
START_LON = 74.3587


# --------------------------------------------------
# FIND HOSPITAL
# --------------------------------------------------

matches = find_hospital_by_name("PKLI")


if not matches:

    print("Hospital not found.")

    exit()


hospital = matches[0]


# --------------------------------------------------
# GET ROUTES
# --------------------------------------------------

print("Calculating routes...")

routes = get_routes(
    START_LAT,
    START_LON,
    hospital["latitude"],
    hospital["longitude"]
)


if not routes:

    print("No routes found.")

    exit()


print(
    f"{len(routes)} route(s) received."
)


# --------------------------------------------------
# CREATE MAP
# --------------------------------------------------

print("Creating map...")

route_map = create_route_map(
    START_LAT,
    START_LON,
    hospital,
    routes
)


# --------------------------------------------------
# SAVE MAP
# --------------------------------------------------

output_file = (
    "problem2_lahore_routes.html"
)

route_map.save(output_file)


print("\nMap created successfully.")

print(
    f"Saved as: {output_file}"
)