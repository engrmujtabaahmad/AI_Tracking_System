import os
import sqlite3
import streamlit as st
from PIL import Image


# ============================================================
# SMART TRAFFIC MONITORING SYSTEM
# AI-BASED PERSON RE-IDENTIFICATION DASHBOARD
# ============================================================


# ============================================================
# PATH CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DB_PATH = os.path.join(
    BASE_DIR,
    "plate_database",
    "traffic_tracking.db"
)

ASSETS_DIR = os.path.join(
    BASE_DIR,
    "assets"
)

REID_EVIDENCE_DIR = os.path.join(
    ASSETS_DIR,
    "reid_evidence"
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Smart Traffic Monitoring System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 32px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .sub-title {
        font-size: 17px;
        color: #aaaaaa;
        margin-bottom: 25px;
    }

    .metric-box {
        padding: 18px;
        border-radius: 10px;
        background-color: #151a21;
        border: 1px solid #2b313a;
    }

    .success-box {
        padding: 15px;
        border-radius: 8px;
        background-color: #0d4026;
        border: 1px solid #1e7a48;
        color: white;
        margin-bottom: 20px;
    }

    .location-box {
        padding: 18px;
        border-radius: 8px;
        background-color: #12304d;
        border: 1px solid #235b87;
        margin-bottom: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# DATABASE CONNECTION
# ============================================================

def get_connection():

    if not os.path.exists(DB_PATH):
        return None

    conn = sqlite3.connect(DB_PATH)

    conn.row_factory = sqlite3.Row

    return conn


# ============================================================
# GET ALL RE-ID SIGHTINGS
# ============================================================

def get_sightings(reference_image):

    conn = get_connection()

    if conn is None:
        return []

    cursor = conn.cursor()

    query = """
        SELECT
            id,
            reference_image,
            camera_id,
            city,
            road,
            video_source,
            first_seen_timestamp,
            first_seen_frame,
            last_seen_timestamp,
            last_seen_frame,
            best_similarity,
            best_yolo_confidence,
            matching_frames,
            created_at
        FROM person_reid_sightings
        WHERE reference_image = ?
        ORDER BY best_similarity DESC
    """

    cursor.execute(
        query,
        (reference_image,)
    )

    results = cursor.fetchall()

    conn.close()

    return results


# ============================================================
# GET MOST RECENT SIGHTINGS
# ============================================================

def get_all_sightings():

    conn = get_connection()

    if conn is None:
        return []

    cursor = conn.cursor()

    query = """
        SELECT
            id,
            reference_image,
            camera_id,
            city,
            road,
            video_source,
            first_seen_timestamp,
            first_seen_frame,
            last_seen_timestamp,
            last_seen_frame,
            best_similarity,
            best_yolo_confidence,
            matching_frames,
            created_at
        FROM person_reid_sightings
        ORDER BY id DESC
    """

    cursor.execute(query)

    results = cursor.fetchall()

    conn.close()

    return results


# ============================================================
# FIND EVIDENCE IMAGE
# ============================================================

def find_evidence_image(camera_id, frame, similarity):

    if not os.path.exists(REID_EVIDENCE_DIR):
        return None

    similarity_text = f"{similarity:.3f}"

    filename = (
        f"{camera_id}_frame_{frame}"
        f"_similarity_{similarity_text}.jpg"
    )

    path = os.path.join(
        REID_EVIDENCE_DIR,
        filename
    )

    if os.path.exists(path):
        return path

    # --------------------------------------------------------
    # Fallback: search directory
    # --------------------------------------------------------

    try:

        for filename in os.listdir(REID_EVIDENCE_DIR):

            if (
                camera_id in filename
                and f"frame_{frame}" in filename
            ):

                return os.path.join(
                    REID_EVIDENCE_DIR,
                    filename
                )

    except Exception:
        pass

    return None


# ============================================================
# FIND PERSON CROP
# ============================================================

def find_person_crop(camera_id, frame):

    if not os.path.exists(REID_EVIDENCE_DIR):
        return None

    expected_filename = (
        f"{camera_id}_person_frame_{frame}.jpg"
    )

    expected_path = os.path.join(
        REID_EVIDENCE_DIR,
        expected_filename
    )

    if os.path.exists(expected_path):
        return expected_path

    # --------------------------------------------------------
    # Fallback search
    # --------------------------------------------------------

    try:

        for filename in os.listdir(REID_EVIDENCE_DIR):

            if (
                camera_id in filename
                and "person" in filename.lower()
                and f"frame_{frame}" in filename
            ):

                return os.path.join(
                    REID_EVIDENCE_DIR,
                    filename
                )

    except Exception:
        pass

    return None


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🚦 Person Search")

st.sidebar.markdown(
    """
    Search the stored Re-ID database.

    The dashboard does **not** run YOLO or OSNet during search.

    Re-ID processing should be performed separately using:

    `person_reid.py`
    """
)

st.sidebar.divider()


uploaded_file = st.sidebar.file_uploader(
    "Upload reference person image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


reference_image_name = None

uploaded_image = None


if uploaded_file is not None:

    uploaded_image = Image.open(
        uploaded_file
    )

    reference_image_name = uploaded_file.name

    st.sidebar.image(
        uploaded_image,
        caption="Reference Person",
        use_container_width=True
    )


st.sidebar.divider()


search_button = st.sidebar.button(
    "🔎 SEARCH DATABASE",
    use_container_width=True,
    type="primary"
)


refresh_button = st.sidebar.button(
    "🔄 REFRESH RESULTS",
    use_container_width=True
)


# ============================================================
# SESSION STATE
# ============================================================

if "searched_reference" not in st.session_state:

    st.session_state.searched_reference = None


if search_button:

    if uploaded_file is None:

        st.sidebar.error(
            "Please upload a reference image first."
        )

    else:

        st.session_state.searched_reference = (
            uploaded_file.name
        )


if refresh_button:

    st.rerun()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🚦 Smart Traffic Monitoring System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'AI-Based Person Re-Identification Dashboard'
    '</div>',
    unsafe_allow_html=True
)

st.caption(
    "YOLO + OSNet Re-ID | SQLite Database | Evidence Management"
)


# ============================================================
# DATABASE CHECK
# ============================================================

if not os.path.exists(DB_PATH):

    st.error(
        "Traffic tracking database was not found."
    )

    st.code(DB_PATH)

    st.stop()


# ============================================================
# DETERMINE SEARCH IMAGE
# ============================================================

searched_reference = (
    st.session_state.searched_reference
)


# ============================================================
# IF SEARCHING SPECIFIC PERSON
# ============================================================

if searched_reference:

    results = get_sightings(
        searched_reference
    )

    # --------------------------------------------------------
    # SEARCH RESULT
    # --------------------------------------------------------

    st.divider()

    st.subheader("🔎 Search Result")

    if not results:

        st.error(
            f"No database record found for `{searched_reference}`."
        )

        st.info(
            "Make sure person_reid.py was executed using "
            "this reference image before searching."
        )

        st.stop()


    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    total_sightings = len(results)

    cameras = len(
        set(
            row["camera_id"]
            for row in results
        )
    )

    best_result = results[0]

    best_similarity = (
        best_result["best_similarity"]
    )

    best_camera = (
        best_result["camera_id"]
    )


    st.markdown(
        """
        <div class="success-box">
        🟢 PERSON FOUND in stored camera results
        </div>
        """,
        unsafe_allow_html=True
    )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Camera Sightings",
            total_sightings
        )


    with col2:

        st.metric(
            "Cameras",
            cameras
        )


    with col3:

        st.metric(
            "Best Similarity",
            f"{best_similarity:.3f}"
        )


    with col4:

        st.metric(
            "Best Camera",
            best_camera
        )


    # ========================================================
    # BEST MATCH
    # ========================================================

    st.divider()

    st.subheader("📍 Best Match")


    left, right = st.columns(2)


    with left:

        st.markdown("### Camera Information")

        st.write(
            f"**Camera:** {best_result['camera_id']}"
        )

        st.write(
            f"**City:** {best_result['city']}"
        )

        st.write(
            f"**Road:** {best_result['road']}"
        )

        st.write(
            f"**Video:** {best_result['video_source']}"
        )


    with right:

        st.markdown("### Detection Information")

        st.write(
            f"**First Seen:** "
            f"{best_result['first_seen_timestamp']}"
        )

        st.write(
            f"**First Frame:** "
            f"{best_result['first_seen_frame']}"
        )

        st.write(
            f"**Last Seen:** "
            f"{best_result['last_seen_timestamp']}"
        )

        st.write(
            f"**Last Frame:** "
            f"{best_result['last_seen_frame']}"
        )

        st.write(
            f"**Similarity:** "
            f"{best_result['best_similarity']:.3f}"
        )

        st.write(
            f"**YOLO Confidence:** "
            f"{best_result['best_yolo_confidence']:.3f}"
        )

        st.write(
            f"**Matching Frames:** "
            f"{best_result['matching_frames']}"
        )


    # ========================================================
    # EVIDENCE
    # ========================================================

    st.divider()

    st.subheader("🖼️ Re-ID Evidence")


    evidence_path = find_evidence_image(
        best_result["camera_id"],
        best_result["first_seen_frame"],
        best_result["best_similarity"]
    )


    crop_path = find_person_crop(
        best_result["camera_id"],
        best_result["first_seen_frame"]
    )


    evidence_col, crop_col = st.columns(2)


    with evidence_col:

        st.markdown("### Evidence Frame")

        if evidence_path:

            st.image(
                evidence_path,
                caption=(
                    f"{best_result['camera_id']} "
                    f"| Frame {best_result['first_seen_frame']} "
                    f"| Similarity "
                    f"{best_result['best_similarity']:.3f}"
                ),
                use_container_width=True
            )

        else:

            st.warning(
                "Evidence frame was not found."
            )


    with crop_col:

        st.markdown("### Matched Person")

        if crop_path:

            st.image(
                crop_path,
                caption="Detected Person Crop",
                use_container_width=True
            )

        else:

            st.warning(
                "Person crop was not found."
            )


    # ========================================================
    # PERSON SIGHTINGS TABLE
    # ========================================================

    st.divider()

    st.subheader("👤 Person Sightings")


    table_data = []


    for row in results:

        table_data.append({

            "Reference Image":
                row["reference_image"],

            "Camera":
                row["camera_id"],

            "City":
                row["city"],

            "Road":
                row["road"],

            "Video":
                row["video_source"],

            "First Seen":
                row["first_seen_timestamp"],

            "First Frame":
                row["first_seen_frame"],

            "Last Seen":
                row["last_seen_timestamp"],

            "Last Frame":
                row["last_seen_frame"],

            "Similarity":
                round(
                    row["best_similarity"],
                    3
                ),

            "YOLO Confidence":
                round(
                    row["best_yolo_confidence"],
                    3
                ),

            "Matching Frames":
                row["matching_frames"]

        })


    st.dataframe(
        table_data,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # DETECTED LOCATIONS
    # ========================================================

    st.divider()

    st.subheader("📌 Detected Locations")


    for index, row in enumerate(
        results,
        start=1
    ):

        st.markdown(
            f"""
            <div class="location-box">

            <b>Sighting #{index}</b><br><br>

            Camera: {row['camera_id']}<br>
            City: {row['city']}<br>
            Road: {row['road']}<br>
            Video: {row['video_source']}<br>
            First Seen: {row['first_seen_timestamp']}<br>
            Last Seen: {row['last_seen_timestamp']}<br>
            Similarity: {row['best_similarity']:.3f}

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# NO SEARCH SELECTED
# ============================================================

else:

    st.divider()

    st.subheader("👤 Person Re-ID Database")


    st.info(
        "Upload a reference person image from the sidebar "
        "and click SEARCH DATABASE."
    )


    # --------------------------------------------------------
    # EXISTING DATABASE STATISTICS
    # --------------------------------------------------------

    all_results = get_all_sightings()


    if all_results:

        st.success(
            f"{len(all_results)} stored Re-ID sighting(s) "
            "available in the database."
        )


        # ----------------------------------------------------
        # DATABASE OVERVIEW
        # ----------------------------------------------------

        col1, col2, col3 = st.columns(3)


        unique_cameras = len(
            set(
                row["camera_id"]
                for row in all_results
            )
        )


        unique_people = len(
            set(
                row["reference_image"]
                for row in all_results
            )
        )


        highest_similarity = max(
            row["best_similarity"]
            for row in all_results
        )


        with col1:

            st.metric(
                "Stored Sightings",
                len(all_results)
            )


        with col2:

            st.metric(
                "Reference Images",
                unique_people
            )


        with col3:

            st.metric(
                "Highest Similarity",
                f"{highest_similarity:.3f}"
            )


        # ----------------------------------------------------
        # STORED RESULTS
        # ----------------------------------------------------

        st.divider()

        st.subheader(
            "📊 Stored Re-ID Results"
        )


        overview_data = []


        for row in all_results:

            overview_data.append({

                "Reference":
                    row["reference_image"],

                "Camera":
                    row["camera_id"],

                "City":
                    row["city"],

                "Road":
                    row["road"],

                "Video":
                    row["video_source"],

                "Similarity":
                    round(
                        row["best_similarity"],
                        3
                    ),

                "First Seen":
                    row["first_seen_timestamp"],

                "Last Seen":
                    row["last_seen_timestamp"]

            })


        st.dataframe(
            overview_data,
            use_container_width=True,
            hide_index=True
        )


    else:

        st.warning(
            "No Re-ID sightings are currently stored."
        )

        st.markdown(
            """
            ### What to do next

            1. Put the reference image in `assets`.
            2. Run `person_reid.py`.
            3. Let YOLO + OSNet search the cameras.
            4. Run `person_reid_evidence.py`.
            5. Open this dashboard.
            6. Upload the same reference image.
            7. Click **SEARCH DATABASE**.
            """
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Smart Traffic Monitoring System | "
    "YOLO + OSNet Person Re-Identification | "
    "SQLite + Streamlit"
)