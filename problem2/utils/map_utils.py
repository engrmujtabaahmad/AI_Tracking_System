import folium
from branca.element import Element


# ============================================================
# ROUTE COLOR PALETTE
#
# Route #1 (recommended / fastest) always gets the primary
# brand blue and is drawn thicker + solid. Any additional
# alternate routes cycle through the rest of this palette and
# are drawn thinner + dashed, so the map never looks like a
# tangle of identical blue lines.
# ============================================================

ROUTE_COLORS = [
    "#2563eb",  # brand blue      - route 1 / recommended
    "#7c3aed",  # violet          - route 2
    "#0891b2",  # teal            - route 3
    "#ea580c",  # orange          - route 4
    "#65a30d",  # olive           - route 5
]


# ============================================================
# CREATE ROUTE MAP
# ============================================================

def create_route_map(
    start_lat,
    start_lon,
    hospital,
    routes
):
    """
    Create an interactive emergency route map.

    Shows:
    - Accident / searched location (RED marker)
    - Nearest hospital (GREEN marker)
    - All calculated routes (BLUE / accent-colored lines)
    - A legend explaining what each marker/line color means
    - Route distance and estimated travel time
    """

    # ========================================================
    # HOSPITAL INFORMATION
    # ========================================================

    hospital_lat = hospital["latitude"]

    hospital_lon = hospital["longitude"]

    hospital_name = hospital.get(
        "name",
        "Nearest Hospital"
    )

    hospital_address = hospital.get(
        "address",
        "Address not available"
    )

    # ========================================================
    # CREATE MAP
    # ========================================================

    center_lat = (
        start_lat + hospital_lat
    ) / 2

    center_lon = (
        start_lon + hospital_lon
    ) / 2

    route_map = folium.Map(
        location=[
            center_lat,
            center_lon
        ],
        zoom_start=14,
        width="100%",
        height="100%",
        control_scale=True,
        tiles="OpenStreetMap"
    )

    # ========================================================
    # ACCIDENT / SEARCHED LOCATION MARKER  (RED)
    # ========================================================

    accident_popup = f"""
    <div style="
        font-family: 'Inter', Arial, sans-serif;
        min-width: 220px;
    ">

        <h4 style="
            margin: 0 0 8px 0;
            color: #dc2626;
        ">
            🚨 Accident / Searched Location
        </h4>

        <b>Latitude:</b>
        {start_lat}

        <br>

        <b>Longitude:</b>
        {start_lon}

    </div>
    """

    folium.Marker(
        location=[
            start_lat,
            start_lon
        ],

        popup=folium.Popup(
            accident_popup,
            max_width=300
        ),

        tooltip="🚨 Accident / Searched Location",

        icon=folium.Icon(
            color="red",
            icon="exclamation-triangle",
            prefix="fa"
        )
    ).add_to(route_map)

    # ========================================================
    # HOSPITAL MARKER  (GREEN)
    # ========================================================

    hospital_popup = f"""
    <div style="
        font-family: 'Inter', Arial, sans-serif;
        min-width: 240px;
    ">

        <h4 style="
            margin: 0 0 8px 0;
            color: #059669;
        ">
            🏥 {hospital_name}
        </h4>

        <b>Address:</b>
        <br>

        {hospital_address}

        <br><br>

        <b>Straight-line Distance:</b>
        {hospital.get(
            "distance_km",
            "N/A"
        )} km

    </div>
    """

    folium.Marker(
        location=[
            hospital_lat,
            hospital_lon
        ],

        popup=folium.Popup(
            hospital_popup,
            max_width=350
        ),

        tooltip=f"🏥 {hospital_name}",

        icon=folium.Icon(
            color="green",
            icon="plus",
            prefix="fa"
        )
    ).add_to(route_map)

    # ========================================================
    # ROUTES
    # ========================================================

    for index, route in enumerate(
        routes,
        start=1
    ):

        geometry = route.get(
            "geometry"
        )

        if not geometry:
            continue

        coordinates = [
            [
                point[1],
                point[0]
            ]

            for point in geometry.get(
                "coordinates",
                []
            )
        ]

        if not coordinates:
            continue

        # ----------------------------------------------------
        # ROUTE NUMBER
        # ----------------------------------------------------

        route_number = route.get(
            "route_number",
            index
        )

        # ----------------------------------------------------
        # DISTANCE / TIME
        # ----------------------------------------------------

        distance = route.get(
            "distance_km",
            0
        )

        duration = route.get(
            "duration_minutes",
            0
        )

        # ----------------------------------------------------
        # IS THIS THE RECOMMENDED (FIRST) ROUTE?
        # ----------------------------------------------------

        is_primary = (
            route_number == 1
        )

        route_color = ROUTE_COLORS[
            (route_number - 1) % len(ROUTE_COLORS)
        ]

        route_label = (
            "⭐ Recommended Route"
            if is_primary
            else f"Alternate Route {route_number}"
        )

        # ----------------------------------------------------
        # ROUTE POPUP
        # ----------------------------------------------------

        popup_text = f"""
        <div style="
            font-family: 'Inter', Arial, sans-serif;
            min-width: 230px;
        ">

            <h4 style="
                margin: 0 0 8px 0;
                color: {route_color};
            ">
                🚑 {route_label}
            </h4>

            <b>Road Distance:</b>
            {distance} km

            <br>

            <b>Estimated Travel Time:</b>
            {duration} minutes

        </div>
        """

        # ----------------------------------------------------
        # DRAW ROUTE
        #
        # The recommended route is thicker and solid so it
        # stands out. Alternate routes are thinner and dashed
        # so they read as secondary options, not clutter.
        # ----------------------------------------------------

        folium.PolyLine(

            locations=coordinates,

            tooltip=(
                f"🚑 {route_label} | "
                f"{distance} km | "
                f"{duration} min"
            ),

            popup=folium.Popup(
                popup_text,
                max_width=350
            ),

            color=route_color,

            weight=7 if is_primary else 4,

            opacity=0.9 if is_primary else 0.65,

            dash_array=None if is_primary else "10, 8"

        ).add_to(route_map)

    # ========================================================
    # FIT MAP TO START + HOSPITAL
    # ========================================================

    route_map.fit_bounds(
        [
            [
                start_lat,
                start_lon
            ],

            [
                hospital_lat,
                hospital_lon
            ]
        ],

        padding=(
            50,
            50
        )
    )

    # ========================================================
    # LEGEND  (floating card in the bottom-left corner)
    # ========================================================

    legend_html = """
    <div style="
        position: fixed;
        bottom: 24px;
        left: 12px;
        z-index: 9999;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 12px 16px;
        box-shadow: 0 6px 18px -6px rgba(15, 23, 42, 0.25);
        font-family: 'Inter', Arial, sans-serif;
        font-size: 12.5px;
        color: #0f172a;
        line-height: 1.9;
    ">
        <div style="font-weight:700; margin-bottom:4px; font-size:12px;
                    text-transform:uppercase; letter-spacing:0.04em;
                    color:#64748b;">
            Map Legend
        </div>

        <div>
            <span style="display:inline-block; width:11px; height:11px;
                        border-radius:50%; background:#dc2626;
                        margin-right:8px;"></span>
            Accident / Searched Location
        </div>

        <div>
            <span style="display:inline-block; width:11px; height:11px;
                        border-radius:50%; background:#059669;
                        margin-right:8px;"></span>
            Nearest Hospital
        </div>

        <div>
            <span style="display:inline-block; width:16px; height:4px;
                        background:#2563eb; margin-right:6px;
                        vertical-align:middle; border-radius:2px;"></span>
            Recommended Route
        </div>
    </div>
    """

    route_map.get_root().html.add_child(
        Element(legend_html)
    )

    # ========================================================
    # MAP CSS
    # ========================================================

    css = """
    <style>

        html,
        body {

            width: 100%;
            height: 100%;

            margin: 0;
            padding: 0;

        }

        .folium-map {

            width: 100% !important;
            height: 100% !important;

        }

        #map {

            width: 100% !important;
            height: 100% !important;

        }

        .leaflet-container {

            width: 100%;
            height: 100%;

        }

        .leaflet-popup-content-wrapper {

            border-radius: 10px;

        }

    </style>
    """

    route_map.get_root().header.add_child(
        Element(css)
    )

    # ========================================================
    # RETURN MAP
    # ========================================================

    return route_map