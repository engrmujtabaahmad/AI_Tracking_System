import requests


# ============================================================
# OSRM ROUTING SERVER
# ============================================================

OSRM_URL = "https://router.project-osrm.org/route/v1/driving"


# ============================================================
# CALCULATE ROUTES
# ============================================================

def calculate_routes(
    start_latitude,
    start_longitude,
    destination_latitude,
    destination_longitude
):
    """
    Calculate alternative driving routes between
    start location and destination.

    Returns routes containing:
        - route_number
        - distance
        - duration
        - geometry
    """

    coordinates = (
        f"{start_longitude},{start_latitude};"
        f"{destination_longitude},{destination_latitude}"
    )

    url = f"{OSRM_URL}/{coordinates}"

    params = {
        "alternatives": "true",
        "steps": "true",
        "overview": "full",
        "geometries": "geojson"
    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=30
        )

        response.raise_for_status()

        data = response.json()

    except requests.RequestException as e:

        raise RuntimeError(
            f"OSRM routing request failed: {e}"
        )


    if data.get("code") != "Ok":

        raise RuntimeError(
            "Routing failed: "
            + str(
                data.get(
                    "message",
                    "Unknown routing error"
                )
            )
        )


    routes = []


    # ========================================================
    # PROCESS ROUTES
    # ========================================================

    for index, route in enumerate(
        data.get("routes", []),
        start=1
    ):

        distance_km = (
            route["distance"] / 1000
        )

        duration_minutes = (
            route["duration"] / 60
        )


        routes.append(
            {
                "route_number": index,

                "distance_km": round(
                    distance_km,
                    2
                ),

                "duration_minutes": round(
                    duration_minutes,
                    2
                ),

                "geometry": route.get(
                    "geometry"
                ),

                "summary": route.get(
                    "summary",
                    f"Route {index}"
                )
            }
        )


    return routes


# ============================================================
# FASTEST ROUTE
# ============================================================

def get_fastest_route(routes):

    if not routes:

        return None


    return min(
        routes,
        key=lambda route:
        route["duration_minutes"]
    )


# ============================================================
# SHORTEST ROUTE
# ============================================================

def get_shortest_route(routes):

    if not routes:

        return None


    return min(
        routes,
        key=lambda route:
        route["distance_km"]
    )