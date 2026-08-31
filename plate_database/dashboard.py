import streamlit as st
import sqlite3
import pandas as pd
from pathlib import Path

# ============================================================
# SMART TRAFFIC MONITORING SYSTEM
# PROFESSIONAL DASHBOARD
# ============================================================

# ------------------------------------------------------------
# PATHS
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATABASE_PATH = BASE_DIR / "plate_database" / "traffic_tracking.db"
OUTPUT_DIR = BASE_DIR / "plate_detector" / "outputs"


# ------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------

st.set_page_config(
    page_title="Smart Traffic Monitoring System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ------------------------------------------------------------
# CUSTOM CSS
# ------------------------------------------------------------

st.markdown("""
<style>

.main-title {
    font-size: 32px;
    font-weight: 700;
}

.subtitle {
    font-size: 16px;
    color: #9aa0a6;
    margin-bottom: 25px;
}

.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 18px;
    text-align: center;
}

.metric-title {
    font-size: 14px;
    color: #8b949e;
}

.metric-value {
    font-size: 30px;
    font-weight: bold;
    margin-top: 5px;
}

.section-title {
    font-size: 22px;
    font-weight: 600;
    margin-top: 25px;
    margin-bottom: 15px;
}

</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------
# DATABASE CONNECTION
# ------------------------------------------------------------

def get_connection():
    return sqlite3.connect(str(DATABASE_PATH))


# ------------------------------------------------------------
# LOAD DATABASE
# ------------------------------------------------------------

@st.cache_data(ttl=3)
def load_data():

    if not DATABASE_PATH.exists():
        return pd.DataFrame()

    conn = get_connection()

    try:

        query = """
        SELECT
            plate_number,
            camera_id,
            city,
            road,
            timestamp,
            frame,
            confidence,
            video_source
        FROM vehicle_detections
        ORDER BY rowid DESC
        """

        df = pd.read_sql_query(query, conn)

    except Exception as e:

        st.error(f"Database error: {e}")
        df = pd.DataFrame()

    finally:
        conn.close()

    return df


df = load_data()


# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------

st.markdown(
    '<div class="main-title">🚦 Smart Traffic Monitoring System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-Based License Plate Detection, OCR & Vehicle Tracking'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ------------------------------------------------------------
# DATABASE CHECK
# ------------------------------------------------------------

if df.empty:

    st.warning(
        "No vehicle detections are currently available in the database."
    )

    st.info(
        "Run the video processing system first to generate detections."
    )

    st.stop()


# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------

st.sidebar.title("🔎 Search & Filters")

st.sidebar.markdown("### License Plate")

plate_search = st.sidebar.text_input(
    "Enter plate number",
    placeholder="e.g. NI2022"
)

st.sidebar.markdown("### Camera")

camera_options = ["All Cameras"] + sorted(
    df["camera_id"].dropna().unique().tolist()
)

selected_camera = st.sidebar.selectbox(
    "Select Camera",
    camera_options
)

st.sidebar.markdown("### City")

city_options = ["All Cities"] + sorted(
    df["city"].dropna().unique().tolist()
)

selected_city = st.sidebar.selectbox(
    "Select City",
    city_options
)

st.sidebar.markdown("### Road")

road_options = ["All Roads"] + sorted(
    df["road"].dropna().unique().tolist()
)

selected_road = st.sidebar.selectbox(
    "Select Road",
    road_options
)


# ------------------------------------------------------------
# APPLY FILTERS
# ------------------------------------------------------------

filtered_df = df.copy()

if plate_search.strip():

    search_value = plate_search.strip().upper()

    filtered_df = filtered_df[
        filtered_df["plate_number"]
        .astype(str)
        .str.upper()
        .str.replace("-", "", regex=False)
        .str.contains(
            search_value.replace("-", ""),
            na=False
        )
    ]


if selected_camera != "All Cameras":

    filtered_df = filtered_df[
        filtered_df["camera_id"] == selected_camera
    ]


if selected_city != "All Cities":

    filtered_df = filtered_df[
        filtered_df["city"] == selected_city
    ]


if selected_road != "All Roads":

    filtered_df = filtered_df[
        filtered_df["road"] == selected_road
    ]


# ------------------------------------------------------------
# KPI CALCULATIONS
# ------------------------------------------------------------

total_detections = len(filtered_df)

unique_plates = (
    filtered_df["plate_number"].nunique()
    if not filtered_df.empty
    else 0
)

unique_cameras = (
    filtered_df["camera_id"].nunique()
    if not filtered_df.empty
    else 0
)

avg_confidence = (
    filtered_df["confidence"].mean()
    if not filtered_df.empty
    else 0
)


# ------------------------------------------------------------
# KPI CARDS
# ------------------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "🚗 Total Detections",
        total_detections
    )

with col2:

    st.metric(
        "🔢 Unique Plates",
        unique_plates
    )

with col3:

    st.metric(
        "📷 Active Cameras",
        unique_cameras
    )

with col4:

    st.metric(
        "🎯 Avg Confidence",
        f"{avg_confidence:.2f}"
    )


st.divider()


# ============================================================
# VEHICLE SEARCH RESULT
# ============================================================

if plate_search.strip():

    st.markdown(
        '<div class="section-title">🔍 Vehicle Search Result</div>',
        unsafe_allow_html=True
    )

    if filtered_df.empty:

        st.error(
            f"Plate **{plate_search.upper()}** was not found."
        )

    else:

        st.success(
            f"Plate **{plate_search.upper()}** found in database."
        )

        search_col1, search_col2 = st.columns(2)

        latest = filtered_df.iloc[0]

        with search_col1:

            st.markdown("### 🚗 Vehicle Information")

            st.write(
                f"**Plate Number:** {latest['plate_number']}"
            )

            st.write(
                f"**Camera:** {latest['camera_id']}"
            )

            st.write(
                f"**City:** {latest['city']}"
            )

            st.write(
                f"**Road:** {latest['road']}"
            )

        with search_col2:

            st.markdown("### 📍 Last Seen")

            st.write(
                f"**Timestamp:** {latest['timestamp']}"
            )

            st.write(
                f"**Frame:** {latest['frame']}"
            )

            st.write(
                f"**Confidence:** {latest['confidence']:.2f}"
            )

            st.write(
                f"**Video:** {latest['video_source']}"
            )


# ============================================================
# ANALYTICS
# ============================================================

st.markdown(
    '<div class="section-title">📊 Traffic Analytics</div>',
    unsafe_allow_html=True
)

if not filtered_df.empty:

    chart_col1, chart_col2 = st.columns(2)

    # --------------------------------------------------------
    # CAMERA CHART
    # --------------------------------------------------------

    with chart_col1:

        st.markdown("### 📷 Detections by Camera")

        camera_counts = (
            filtered_df["camera_id"]
            .value_counts()
        )

        st.bar_chart(camera_counts)

    # --------------------------------------------------------
    # ROAD CHART
    # --------------------------------------------------------

    with chart_col2:

        st.markdown("### 🛣️ Detections by Road")

        road_counts = (
            filtered_df["road"]
            .value_counts()
        )

        st.bar_chart(road_counts)


# ============================================================
# DETECTION RECORDS
# ============================================================

st.markdown(
    '<div class="section-title">📋 Detection Records</div>',
    unsafe_allow_html=True
)

if filtered_df.empty:

    st.info("No records match the selected filters.")

else:

    display_df = filtered_df.copy()

    display_df["confidence"] = display_df[
        "confidence"
    ].round(2)

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# PLATE HISTORY
# ============================================================

if not filtered_df.empty:

    st.markdown(
        '<div class="section-title">🧭 Vehicle Tracking History</div>',
        unsafe_allow_html=True
    )

    unique_plate_list = sorted(
        filtered_df["plate_number"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_plate = st.selectbox(
        "Select vehicle plate",
        unique_plate_list
    )

    history = filtered_df[
        filtered_df["plate_number"] == selected_plate
    ]

    if not history.empty:

        st.dataframe(
            history[
                [
                    "plate_number",
                    "camera_id",
                    "city",
                    "road",
                    "timestamp",
                    "frame",
                    "confidence",
                    "video_source"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# PROCESSED VIDEO
# ============================================================

st.markdown(
    '<div class="section-title">🎥 Processed Traffic Video</div>',
    unsafe_allow_html=True
)

video_files = []

if OUTPUT_DIR.exists():

    video_files = sorted(
        [
            file for file in OUTPUT_DIR.iterdir()
            if file.suffix.lower() in [
                ".mp4",
                ".avi",
                ".mov",
                ".mkv"
            ]
        ],
        key=lambda x: x.name
    )


if video_files:

    selected_video = st.selectbox(
        "Select processed video",
        video_files,
        format_func=lambda x: x.name
    )

    try:

        video_bytes = selected_video.read_bytes()

        st.video(video_bytes)

    except Exception as e:

        st.error(
            f"Unable to load video: {e}"
        )

else:

    st.info(
        "No processed traffic videos found."
    )


# ============================================================
# SYSTEM INFORMATION
# ============================================================

st.divider()

st.markdown("### ⚙️ System Information")

info_col1, info_col2, info_col3 = st.columns(3)

with info_col1:

    st.write(
        f"**Database:** `{DATABASE_PATH.name}`"
    )

with info_col2:

    st.write(
        f"**Records:** `{len(df)}`"
    )

with info_col3:

    st.write(
        "**AI Pipeline:** YOLO11 + EasyOCR"
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Smart Traffic Monitoring System | "
    "YOLO11 + EasyOCR + SQLite + Streamlit"
)