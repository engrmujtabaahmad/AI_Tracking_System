import os
import sys
import sqlite3
import subprocess
import textwrap

import pandas as pd
import streamlit as st
import requests

from streamlit_folium import st_folium
from streamlit_geolocation import streamlit_geolocation


def html_block(content):
    """
    Render raw HTML safely through st.markdown.

    Streamlit's markdown parser treats any line indented by
    4+ spaces as an indented CODE block, which makes hand-formatted
    HTML show up as literal text instead of rendering. Dedenting
    (and stripping) every line before passing it to st.markdown
    avoids that entirely.
    """

    st.markdown(
        textwrap.dedent(content).strip(),
        unsafe_allow_html=True
    )


# ============================================================
# PROJECT PATHS
# ============================================================

DASHBOARD_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

BASE_DIR = os.path.dirname(
    DASHBOARD_DIR
)


DATABASE_PATH = os.path.join(
    BASE_DIR,
    "plate_database",
    "traffic_tracking.db"
)


PERSON_REID_SCRIPT = os.path.join(
    BASE_DIR,
    "plate_database",
    "person_reid.py"
)


UPLOAD_DIR = os.path.join(
    BASE_DIR,
    "assets",
    "uploaded_references"
)


ASSETS_DIR = os.path.join(
    BASE_DIR,
    "assets"
)


VIDEO_EXTENSIONS = (
    ".mp4",
    ".avi",
    ".mov",
    ".mkv"
)


os.makedirs(
    UPLOAD_DIR,
    exist_ok=True
)


os.makedirs(
    ASSETS_DIR,
    exist_ok=True
)


# ============================================================
# IMPORT PROJECT SERVICES
# ============================================================

sys.path.insert(
    0,
    BASE_DIR
)


from problem2.services.hospital_service import (
    find_nearest_hospitals
)


from problem2.services.route_service import (
    calculate_routes,
    get_fastest_route,
    get_shortest_route
)


from problem2.utils.map_utils import (
    create_route_map
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Accident Response System",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS  —  PROFESSIONAL THEME
# ============================================================

st.markdown(
    """
    <style>

    /* =====================================================
       DESIGN SYSTEM - "DISPATCH CONSOLE"
       A control-room aesthetic for an AI traffic / accident
       monitoring system: dark panels, amber/red alert accent,
       cyan live-data accent, monospace data readouts, and a
       subtle camera-viewfinder corner-bracket motif tying the
       UI back to the detection/tracking subject matter.
       ===================================================== */

    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    :root {
        --bg-void: #0a0e16;
        --bg-panel: #121826;
        --bg-panel-raised: #1a2233;
        --border-line: #26314a;
        --border-line-strong: #34405c;
        --text-primary: #eef1f7;
        --text-secondary: #8b96ac;
        --text-muted: #5f6b82;
        --accent-amber: #f5a623;
        --accent-amber-soft: rgba(245, 166, 35, 0.14);
        --accent-red: #ef4444;
        --accent-red-soft: rgba(239, 68, 68, 0.14);
        --accent-cyan: #22d3ee;
        --accent-cyan-soft: rgba(34, 211, 238, 0.12);
        --accent-green: #34d399;
        --accent-green-soft: rgba(52, 211, 153, 0.14);
    }

    /* ---------- Base App Background ---------- */
    .stApp {
        background:
            radial-gradient(circle at 12% -10%, rgba(34, 211, 238, 0.07), transparent 42%),
            radial-gradient(circle at 90% 110%, rgba(245, 166, 35, 0.06), transparent 45%),
            var(--bg-void);
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* ---------- Header ---------- */
    .app-header {
        background: linear-gradient(135deg, #0c1120 0%, #131a2c 55%, #1a2338 100%);
        border: 1px solid var(--border-line);
        border-radius: 16px;
        padding: 32px 38px;
        margin-bottom: 26px;
        position: relative;
        overflow: hidden;
    }

    .app-header::before {
        content: "";
        position: absolute;
        inset: 0;
        background-image: repeating-linear-gradient(
            0deg,
            rgba(255,255,255,0.015) 0px,
            rgba(255,255,255,0.015) 1px,
            transparent 1px,
            transparent 3px
        );
        pointer-events: none;
    }

    .app-header::after {
        content: "";
        position: absolute;
        top: 18px;
        right: 18px;
        width: 22px;
        height: 22px;
        border-top: 2px solid var(--accent-cyan);
        border-right: 2px solid var(--accent-cyan);
        opacity: 0.55;
    }

    .app-eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        color: var(--accent-cyan);
        font-family: 'IBM Plex Mono', monospace;
        font-size: 12px;
        font-weight: 500;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .app-eyebrow .dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: var(--accent-green);
        box-shadow: 0 0 0 4px rgba(52, 211, 153, 0.20);
        animation: pulse-dot 2.2s ease-in-out infinite;
    }

    @keyframes pulse-dot {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }

    .main-title {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 33px;
        font-weight: 700;
        color: var(--text-primary);
        letter-spacing: -0.01em;
        margin: 0 0 6px 0;
        line-height: 1.2;
    }

    .subtitle {
        color: var(--text-secondary);
        font-family: 'IBM Plex Mono', monospace;
        font-size: 13px;
        letter-spacing: 0.01em;
        margin: 0;
    }

    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #080b12 0%, #0b0f18 100%);
        border-right: 1px solid var(--border-line);
    }

    section[data-testid="stSidebar"] * {
        color: var(--text-secondary) !important;
    }

    section[data-testid="stSidebar"] h1 {
        font-family: 'Space Grotesk', sans-serif;
        font-size: 19px;
        font-weight: 700;
        color: var(--text-primary) !important;
        padding-bottom: 4px;
        border-bottom: 1px solid var(--border-line);
        margin-bottom: 14px;
    }

    /* ---------- Section Headers ---------- */
    h1, h2, h3 {
        color: var(--text-primary) !important;
        font-family: 'Space Grotesk', sans-serif !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em;
    }

    h2 {
        font-size: 23px !important;
        border-left: 3px solid var(--accent-cyan);
        padding-left: 12px;
        margin-top: 8px !important;
    }

    h3 {
        font-size: 17px !important;
        color: var(--text-primary) !important;
    }

    /* ---------- Buttons ---------- */
    .stButton > button {
        background: linear-gradient(120deg, var(--accent-red) 0%, var(--accent-amber) 100%);
        color: #0a0e16;
        border: none;
        border-radius: 10px;
        padding: 11px 24px;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 14.5px;
        letter-spacing: 0.01em;
        box-shadow: 0 8px 20px -8px rgba(239, 68, 68, 0.45);
        transition: transform 0.12s ease, box-shadow 0.12s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 10px 26px -8px rgba(245, 166, 35, 0.55);
    }

    .stButton > button p {
        color: #0a0e16 !important;
        font-weight: 700 !important;
    }

    /* ---------- Inputs ---------- */
    .stTextInput > div > div input,
    .stNumberInput > div > div input,
    .stTextArea textarea {
        border-radius: 10px !important;
        border: 1.5px solid var(--border-line) !important;
        padding: 10px 14px !important;
        font-size: 14.5px !important;
        background: var(--bg-panel) !important;
        color: var(--text-primary) !important;
        caret-color: var(--accent-cyan) !important;
        font-family: 'IBM Plex Mono', monospace !important;
    }

    .stTextInput > div > div input::placeholder,
    .stTextArea textarea::placeholder {
        color: var(--text-muted) !important;
        opacity: 1 !important;
    }

    .stTextInput > div > div input:focus,
    .stNumberInput > div > div input:focus,
    .stTextArea textarea:focus {
        border-color: var(--accent-cyan) !important;
        box-shadow: 0 0 0 3px var(--accent-cyan-soft) !important;
    }

    /* Labels above inputs - console-style eyebrow labels */
    .stTextInput label p,
    .stNumberInput label p,
    .stTextArea label p,
    .stFileUploader label p,
    label[data-testid="stWidgetLabel"] p {
        color: var(--text-secondary) !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-weight: 600 !important;
        font-size: 12px !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    .stCaptionContainer, [data-testid="stCaptionContainer"] {
        color: var(--text-muted) !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 12px !important;
    }

    /* File uploader dropzone */
    section[data-testid="stFileUploaderDropzone"] {
        background: var(--bg-panel) !important;
        border: 1.5px dashed var(--border-line) !important;
        border-radius: 12px !important;
    }

    section[data-testid="stFileUploaderDropzone"] * {
        color: var(--text-secondary) !important;
    }

    .main p, .main span, .main li, .main label {
        color: var(--text-primary);
    }

    /* ---------- Metrics ---------- */
    div[data-testid="stMetric"] {
        background: var(--bg-panel);
        border: 1px solid var(--border-line);
        border-radius: 12px;
        padding: 16px 18px;
        position: relative;
    }

    div[data-testid="stMetricLabel"] {
        color: var(--text-secondary) !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-weight: 600 !important;
        font-size: 11.5px !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }

    div[data-testid="stMetricValue"] {
        color: var(--accent-cyan) !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-weight: 700 !important;
    }

    /* ---------- Corner-bracket signature (viewfinder motif) ---------- */
    .section-card, div[data-testid="stMetric"] {
        position: relative;
    }

    .section-card::before, div[data-testid="stMetric"]::before,
    .section-card::after,  div[data-testid="stMetric"]::after {
        content: "";
        position: absolute;
        width: 13px;
        height: 13px;
        border-color: var(--accent-cyan);
        opacity: 0.5;
        pointer-events: none;
    }

    .section-card::before, div[data-testid="stMetric"]::before {
        top: -1px; left: -1px;
        border-top: 2px solid;
        border-left: 2px solid;
        border-top-left-radius: 4px;
    }

    .section-card::after, div[data-testid="stMetric"]::after {
        bottom: -1px; right: -1px;
        border-bottom: 2px solid;
        border-right: 2px solid;
        border-bottom-right-radius: 4px;
    }

    /* ---------- Alerts (color-coded by type) ---------- */
    div[data-testid="stAlert"] {
        border-radius: 10px;
        border: 1px solid var(--border-line);
        background: var(--bg-panel);
        padding: 4px 6px;
    }

    div[data-testid="stAlert"]:has(div[data-testid="stAlertContentSuccess"]) {
        border-left: 3px solid var(--accent-green) !important;
    }

    div[data-testid="stAlert"]:has(div[data-testid="stAlertContentWarning"]) {
        border-left: 3px solid var(--accent-amber) !important;
    }

    div[data-testid="stAlert"]:has(div[data-testid="stAlertContentError"]) {
        border-left: 3px solid var(--accent-red) !important;
    }

    div[data-testid="stAlert"]:has(div[data-testid="stAlertContentInfo"]) {
        border-left: 3px solid var(--accent-cyan) !important;
    }

    div[data-testid="stAlert"] p,
    div[data-testid="stAlert"] span,
    div[data-testid="stAlert"] li,
    div[data-testid="stAlert"] strong,
    div[data-testid="stAlertContentInfo"] *,
    div[data-testid="stAlertContentSuccess"] *,
    div[data-testid="stAlertContentWarning"] *,
    div[data-testid="stAlertContentError"] * {
        color: var(--text-primary) !important;
    }

    /* ---------- DataFrames / Tables ---------- */
    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid var(--border-line);
    }

    /* ---------- Expander ---------- */
    div[data-testid="stExpander"] {
        border: 1px solid var(--border-line);
        border-radius: 12px;
        background: var(--bg-panel);
    }

    div[data-testid="stExpander"] summary,
    div[data-testid="stExpander"] summary p,
    div[data-testid="stExpander"] * {
        color: var(--text-primary) !important;
    }

    /* ---------- Divider ---------- */
    hr {
        border-color: var(--border-line) !important;
        margin: 22px 0 !important;
    }

    /* ---------- Section Cards ---------- */
    .section-card {
        background: var(--bg-panel);
        border: 1px solid var(--border-line);
        border-radius: 16px;
        padding: 22px 26px;
        margin-bottom: 18px;
    }

    .section-lead {
        color: var(--text-secondary);
        font-family: 'IBM Plex Mono', monospace;
        font-size: 13px;
        margin-bottom: 6px;
    }

    /* ---------- Workflow Steps ---------- */
    .workflow-step {
        display: flex;
        align-items: flex-start;
        gap: 14px;
        padding: 14px 4px;
        border-bottom: 1px solid var(--border-line);
    }

    .workflow-step:last-child {
        border-bottom: none;
    }

    .workflow-num {
        flex-shrink: 0;
        width: 34px;
        height: 34px;
        border-radius: 10px;
        background: linear-gradient(120deg, var(--accent-red) 0%, var(--accent-amber) 100%);
        color: #0a0e16;
        display: flex;
        align-items: center;
        justify-content: center;
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        font-size: 14.5px;
    }

    .workflow-title {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 700;
        color: var(--text-primary);
        font-size: 15px;
        margin-bottom: 2px;
    }

    .workflow-desc {
        color: var(--text-secondary);
        font-size: 13.5px;
    }

    /* ---------- Status Pills ---------- */
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: var(--accent-green-soft);
        color: var(--accent-green);
        border: 1px solid rgba(52, 211, 153, 0.3);
        border-radius: 999px;
        padding: 3px 10px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11.5px;
        font-weight: 600;
    }

    /* ---------- Footer ---------- */
    .app-footer {
        text-align: center;
        color: var(--text-muted);
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11.5px;
        letter-spacing: 0.03em;
        padding-top: 6px;
    }

    /* ---------- st.code blocks ---------- */
    div[data-testid="stCodeBlock"] {
        border-radius: 12px;
        border: 1px solid var(--border-line);
    }

    /* ---------- File uploaded name chips ---------- */
    div[data-testid="stFileUploaderFile"] {
        background: var(--bg-panel-raised) !important;
        border-radius: 8px;
        color: var(--text-primary) !important;
    }

    div[data-testid="stFileUploaderFile"] * {
        color: var(--text-primary) !important;
    }

    /* ---------- Uploaded image caption ---------- */
    div[data-testid="stImage"] figcaption {
        color: var(--text-secondary) !important;
        font-size: 13px !important;
    }

    /* ---------- Spinner text ---------- */
    div[data-testid="stSpinner"] p {
        color: var(--accent-cyan) !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-weight: 600 !important;
    }

    .main [data-testid="stMarkdownContainer"] p,
    .main [data-testid="stMarkdownContainer"] li,
    .main [data-testid="stMarkdownContainer"] strong {
        color: var(--text-primary);
    }

    /* =====================================================
       SEGMENTED RADIO CONTROL
       Replaces Streamlit's default (buggy-contrast) radio
       circles entirely with a filled-pill segmented control.
       Selection state is driven directly off the real native
       <input type="radio"> via :has(), so it can never drift
       out of sync with what's actually selected.
       ===================================================== */

    div[data-testid="stRadio"] [role="radiogroup"] {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 4px;
    }

    div[data-testid="stRadio"] [role="radiogroup"] label {
        display: flex;
        align-items: center;
        gap: 8px;
        background: var(--bg-panel);
        border: 1.5px solid var(--border-line);
        border-radius: 10px;
        padding: 10px 16px;
        cursor: pointer;
        transition: border-color 0.15s ease, background 0.15s ease;
    }

    div[data-testid="stRadio"] [role="radiogroup"] label:hover {
        border-color: var(--accent-cyan);
    }

    /* hide the native circle - the pill fill IS the indicator */
    div[data-testid="stRadio"] [role="radiogroup"] label > div:first-child {
        display: none !important;
    }

    div[data-testid="stRadio"] [role="radiogroup"] label p,
    div[data-testid="stRadio"] [role="radiogroup"] label span,
    div[data-testid="stRadio"] [role="radiogroup"] label div[data-testid="stMarkdownContainer"] p {
        color: var(--text-secondary) !important;
        font-weight: 500 !important;
        font-size: 13.5px !important;
        opacity: 1 !important;
        margin: 0 !important;
    }

    section[data-testid="stSidebar"] div[data-testid="stRadio"] [role="radiogroup"] label {
        background: var(--bg-panel-raised) !important;
        border-color: var(--border-line-strong) !important;
    }

    /* SELECTED option: unmistakable amber fill, dark text */
    div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) {
        background: var(--accent-amber) !important;
        border-color: var(--accent-amber) !important;
    }

    div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) p,
    div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) span,
    div[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked) div[data-testid="stMarkdownContainer"] p {
        color: #0a0e16 !important;
        font-weight: 700 !important;
    }

    /* ---------- GPS location widget ---------- */
    iframe[title*="geolocation"] {
        width: 56px !important;
        height: 56px !important;
        min-width: 56px !important;
        border-radius: 10px !important;
        box-shadow: 0 0 0 2px var(--accent-cyan-soft), 0 4px 14px -4px rgba(34, 211, 238, 0.4) !important;
    }

    .geo-widget-card {
        background: var(--bg-panel);
        border: 1.5px dashed var(--accent-cyan);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 14px;
    }

    .geo-widget-label {
        color: var(--text-primary);
        font-family: 'IBM Plex Mono', monospace;
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.02em;
    }

</style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

html_block(
    """
    <div class="app-header">
        <div class="app-eyebrow"><span class="dot"></span>LIVE SYSTEM &nbsp;·&nbsp; AI-POWERED</div>
        <div class="main-title">🚨 AI Accident Response System</div>
        <p class="subtitle">AI-Based Traffic Monitoring, Location Detection &amp; Emergency Response</p>
    </div>
    """
)


# ============================================================
# SESSION STATE
# ============================================================

if "accident_result" not in st.session_state:

    st.session_state.accident_result = None


if "accident_location" not in st.session_state:

    st.session_state.accident_location = ""


# ============================================================
# DATABASE
# ============================================================

def get_connection():

    return sqlite3.connect(
        DATABASE_PATH
    )


# ============================================================
# VEHICLE SEARCH
# ============================================================

def search_vehicle(
    plate_number
):

    plate_number = (
        plate_number
        .upper()
        .replace(" ", "")
        .replace("-", "")
    )

    conn = get_connection()

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
        WHERE REPLACE(
            REPLACE(
                UPPER(plate_number),
                ' ',
                ''
            ),
            '-',
            ''
        ) = ?
        ORDER BY id DESC
    """

    try:

        df = pd.read_sql_query(
            query,
            conn,
            params=(plate_number,)
        )

    except Exception as e:

        st.error(
            f"Vehicle database error: {e}"
        )

        df = pd.DataFrame()

    conn.close()

    return df


# ============================================================
# PERSON SEARCH
# ============================================================

def get_person_sightings(
    reference_image
):

    conn = get_connection()

    query = """
        SELECT
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

    try:

        df = pd.read_sql_query(
            query,
            conn,
            params=(reference_image,)
        )

    except Exception as e:

        st.error(
            f"Person database error: {e}"
        )

        df = pd.DataFrame()

    conn.close()

    return df


# ============================================================
# RUN PERSON RE-ID
#
# IMPORTANT:
# This version uses Popen instead of subprocess.run().
#
# It allows us to receive the output from person_reid.py
# while the process is still running.
# ============================================================

def run_person_reid(
    image_path,
    output_placeholder
):

    command = [
        sys.executable,
        PERSON_REID_SCRIPT,
        image_path
    ]

    output_lines = []

    try:

        process = subprocess.Popen(

            command,

            cwd=BASE_DIR,

            stdout=subprocess.PIPE,

            stderr=subprocess.STDOUT,

            text=True,

            encoding="utf-8",

            errors="replace",

            bufsize=1
        )


        # ----------------------------------------------------
        # Read output LIVE
        # ----------------------------------------------------

        if process.stdout is not None:

            for line in process.stdout:

                line = line.rstrip()

                if not line:
                    continue

                output_lines.append(
                    line
                )


                # --------------------------------------------
                # Update Streamlit immediately
                # --------------------------------------------

                output_placeholder.code(
                    "\n".join(
                        output_lines
                    ),
                    language="text"
                )


        # ----------------------------------------------------
        # Wait for process to finish
        # ----------------------------------------------------

        process.wait()


        return (

            process.returncode,

            "\n".join(
                output_lines
            ),

            ""

        )


    except Exception as e:

        return (

            -1,

            "\n".join(
                output_lines
            ),

            str(e)

        )


# ============================================================
# CLEAN OLD REFERENCE FILES
# ============================================================

def cleanup_old_reference_files():

    """
    Remove previous search_reference files so that
    old uploaded images do not create confusion.
    """

    if not os.path.exists(
        UPLOAD_DIR
    ):

        return


    for filename in os.listdir(
        UPLOAD_DIR
    ):

        if filename.startswith(
            "search_reference."
        ):

            file_path = os.path.join(
                UPLOAD_DIR,
                filename
            )

            try:

                if os.path.isfile(
                    file_path
                ):

                    os.remove(
                        file_path
                    )

            except Exception:

                pass


# ============================================================
# LOCATION GEOCODING  (address -> coordinates)
# ============================================================

@st.cache_data(
    ttl=300,
    show_spinner=False
)
def geocode_location(
    location_name
):

    """
    Convert a location name/address into
    latitude and longitude using Nominatim.
    """

    location_name = (
        location_name.strip()
    )


    if not location_name:

        return None


    search_query = (
        f"{location_name}, Lahore, Punjab, Pakistan"
    )


    url = (
        "https://nominatim.openstreetmap.org/search"
    )


    params = {

        "q":
            search_query,

        "format":
            "jsonv2",

        "limit":
            1,

        "countrycodes":
            "pk",

        "addressdetails":
            1

    }


    headers = {

        "User-Agent":
            "AI-Accident-Response-System/1.0"

    }


    try:

        response = requests.get(

            url,

            params=params,

            headers=headers,

            timeout=15

        )


        response.raise_for_status()


        results = response.json()


    except requests.RequestException as e:

        raise RuntimeError(
            f"Location search failed: {e}"
        )


    if not results:

        return None


    result = results[0]


    try:

        latitude = float(
            result["lat"]
        )


        longitude = float(
            result["lon"]
        )


    except (
        KeyError,
        ValueError
    ):

        return None


    return {

        "latitude":
            latitude,

        "longitude":
            longitude,

        "display_name":
            result.get(
                "display_name",
                location_name
            )

    }


# ============================================================
# REVERSE GEOCODING  (coordinates -> address)
#
# Used for the GPS "current location" flow - the browser gives
# us raw latitude/longitude, this turns it into a readable address.
# ============================================================

@st.cache_data(
    ttl=300,
    show_spinner=False
)
def reverse_geocode(
    latitude,
    longitude
):

    """
    Convert GPS coordinates into a human-readable address
    using Nominatim.
    """

    url = (
        "https://nominatim.openstreetmap.org/reverse"
    )


    params = {

        "lat":
            latitude,

        "lon":
            longitude,

        "format":
            "jsonv2"

    }


    headers = {

        "User-Agent":
            "AI-Accident-Response-System/1.0"

    }


    try:

        response = requests.get(

            url,

            params=params,

            headers=headers,

            timeout=15

        )


        response.raise_for_status()


        result = response.json()


    except requests.RequestException:

        return f"{latitude}, {longitude}"


    return result.get(
        "display_name",
        f"{latitude}, {longitude}"
    )


# ============================================================
# CALCULATE ACCIDENT RESPONSE  (shared logic)
#
# Used by BOTH the manual text-search flow and the GPS flow.
# "location" must be a dict with latitude, longitude, display_name.
# ============================================================

def calculate_accident_response_from_location(
    location,
    search_label=None
):

    latitude = location["latitude"]

    longitude = location["longitude"]


    # --------------------------------------------------------
    # FIND NEAREST HOSPITALS
    # --------------------------------------------------------

    hospitals = find_nearest_hospitals(

        latitude,

        longitude,

        limit=5

    )


    if not hospitals:

        return {

            "error":
                "No hospitals were found."

        }


    nearest_hospital = hospitals[0]


    # --------------------------------------------------------
    # CALCULATE ROUTES
    # --------------------------------------------------------

    try:

        routes = calculate_routes(

            latitude,

            longitude,

            nearest_hospital[
                "latitude"
            ],

            nearest_hospital[
                "longitude"
            ]

        )

    except Exception as e:

        return {

            "error":
                f"Route calculation failed: {e}"

        }


    if not routes:

        return {

            "error":
                "No driving routes were found."

        }


    # --------------------------------------------------------
    # ADD ROUTE NUMBERS
    # --------------------------------------------------------

    for index, route in enumerate(

        routes,

        start=1

    ):

        route["route_number"] = index


    # --------------------------------------------------------
    # FASTEST ROUTE
    # --------------------------------------------------------

    fastest = get_fastest_route(
        routes
    )


    # --------------------------------------------------------
    # SHORTEST ROUTE
    # --------------------------------------------------------

    shortest = get_shortest_route(
        routes
    )


    # --------------------------------------------------------
    # CREATE MAP
    # --------------------------------------------------------

    route_map = create_route_map(

        latitude,

        longitude,

        nearest_hospital,

        routes

    )


    # --------------------------------------------------------
    # RETURN COMPLETE RESULT
    # --------------------------------------------------------

    return {

        "location":
            location,

        "search_label":
            search_label
            or location.get(
                "display_name",
                f"{latitude}, {longitude}"
            ),

        "hospitals":
            hospitals,

        "nearest_hospital":
            nearest_hospital,

        "routes":
            routes,

        "fastest":
            fastest,

        "shortest":
            shortest,

        "map":
            route_map

    }


# ============================================================
# CALCULATE ACCIDENT RESPONSE  (manual text search entry point)
# ============================================================

def calculate_accident_response(
    location_name
):

    location = geocode_location(
        location_name
    )


    if location is None:

        return {

            "error":
                "Location could not be found. "
                "Try a more specific Lahore location."

        }


    return calculate_accident_response_from_location(

        location,

        search_label=location_name

    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🚨 Control Panel"
)


st.sidebar.write(
    "Select the operation you want to perform."
)


mode = st.sidebar.radio(

    "System Mode",

    [

        "🏠 Dashboard",

        "🚗 Vehicle / Number Plate",

        "👤 Person Search",

        "📹 Upload Video",

        "🚑 Accident Response"

    ]

)


# ============================================================
# DASHBOARD
# ============================================================

if mode == "🏠 Dashboard":

    st.header(
        "🚨 Unified Accident Response Dashboard"
    )


    html_block(
        """
        <p class="section-lead">
        This system combines traffic tracking, person re-identification
        and emergency hospital routing into a single operational view.
        </p>
        """
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "🚗 Traffic Tracking",
            "ACTIVE"
        )


    with col2:

        st.metric(
            "👤 Person Re-ID",
            "ACTIVE"
        )


    with col3:

        st.metric(
            "🏥 Emergency Response",
            "ACTIVE"
        )


    st.divider()


    st.subheader(
        "System Workflow"
    )


    html_block(
        """
        <div class="section-card">
        <div class="workflow-step">
        <div class="workflow-num">1</div>
        <div>
        <div class="workflow-title">Detect</div>
        <div class="workflow-desc">Vehicle or person is detected from camera footage.</div>
        </div>
        </div>
        <div class="workflow-step">
        <div class="workflow-num">2</div>
        <div>
        <div class="workflow-title">Locate</div>
        <div class="workflow-desc">The system determines the exact location of the event.</div>
        </div>
        </div>
        <div class="workflow-step">
        <div class="workflow-num">3</div>
        <div>
        <div class="workflow-title">Find Hospital</div>
        <div class="workflow-desc">The nearest hospital to the location is identified.</div>
        </div>
        </div>
        <div class="workflow-step">
        <div class="workflow-num">4</div>
        <div>
        <div class="workflow-title">Calculate Routes</div>
        <div class="workflow-desc">All available driving routes are calculated automatically.</div>
        </div>
        </div>
        <div class="workflow-step">
        <div class="workflow-num">5</div>
        <div>
        <div class="workflow-title">Recommend</div>
        <div class="workflow-desc">The fastest and shortest routes are displayed for response teams.</div>
        </div>
        </div>
        </div>
        """
    )


# ============================================================
# VEHICLE SEARCH
# ============================================================


elif mode == "🚗 Vehicle / Number Plate":

    st.header(
        "🚗 Vehicle / Number Plate Search"
    )


    html_block(
        '<p class="section-lead">Search the tracking database using a detected number plate.</p>'
    )


    plate = st.text_input(

        "Enter Number Plate",

        placeholder="Example: NI2022"

    )


    search_button = st.button(

        "🔎 SEARCH VEHICLE",

        type="primary"

    )


    if search_button:

        if not plate.strip():

            st.warning(
                "Please enter a number plate."
            )

        else:

            with st.spinner(
                "Searching vehicle database..."
            ):

                df = search_vehicle(
                    plate
                )


            if df.empty:

                st.error(
                    "❌ Vehicle / plate not found."
                )

            else:

                st.success(
                    f"✅ Vehicle found in "
                    f"{len(df)} record(s)."
                )


                st.dataframe(

                    df,

                    use_container_width=True,

                    hide_index=True

                )


                best = df.iloc[0]


                st.subheader(
                    "📍 Latest Known Location"
                )


                col1, col2, col3 = st.columns(3)


                with col1:

                    st.metric(
                        "Camera",
                        best["camera_id"]
                    )


                with col2:

                    st.metric(
                        "City",
                        best["city"]
                    )


                with col3:

                    st.metric(
                        "Road",
                        best["road"]
                    )


                st.info(

                    f"""
                    **Number Plate:** {best['plate_number']}

                    **Camera:** {best['camera_id']}

                    **Location:** {best['city']}

                    **Road:** {best['road']}

                    **Timestamp:** {best['timestamp']}

                    **Video:** {best['video_source']}
                    """

                )


# ============================================================
# PERSON SEARCH
# ============================================================

elif mode == "👤 Person Search":

    st.header(
        "👤 Person Re-Identification"
    )


    html_block(
        '<p class="section-lead">Upload a reference person image and search the available traffic cameras.</p>'
    )


    uploaded_file = st.file_uploader(

        "Upload Person Image",

        type=[

            "jpg",

            "jpeg",

            "png"

        ]

    )


    search_button = st.button(

        "🔎 SEARCH PERSON",

        type="primary"

    )


    if uploaded_file is not None:

        st.image(

            uploaded_file,

            caption="Reference Image",

            width=250

        )


    # ========================================================
    # SEARCH PERSON BUTTON
    # ========================================================

    if search_button:

        if uploaded_file is None:

            st.warning(
                "Please upload a person image first."
            )

        else:

            # ------------------------------------------------
            # Remove old search_reference files
            # ------------------------------------------------

            cleanup_old_reference_files()


            # ------------------------------------------------
            # Determine extension
            # ------------------------------------------------

            extension = os.path.splitext(

                uploaded_file.name

            )[1].lower()


            if extension not in [

                ".jpg",

                ".jpeg",

                ".png"

            ]:

                extension = ".jpg"


            # ------------------------------------------------
            # Save uploaded image
            # ------------------------------------------------

            saved_filename = (
                "search_reference"
                + extension
            )


            saved_path = os.path.join(

                UPLOAD_DIR,

                saved_filename

            )


            with open(

                saved_path,

                "wb"

            ) as file:

                file.write(

                    uploaded_file.getbuffer()

                )


            # ------------------------------------------------
            # Verify image exists
            # ------------------------------------------------

            if not os.path.exists(
                saved_path
            ):

                st.error(
                    "Failed to save reference image."
                )

                st.stop()


            # ------------------------------------------------
            # START SEARCH
            # ------------------------------------------------

            st.info(
                "🔄 Searching all traffic cameras..."
            )


            # ------------------------------------------------
            # LIVE OUTPUT BOX
            # ------------------------------------------------

            progress_box = st.empty()


            # ------------------------------------------------
            # RUN PERSON RE-ID
            # ------------------------------------------------

            with st.spinner(

                "Face Recognition Re-ID is processing..."

            ):

                return_code, output, error = (
                    run_person_reid(

                        saved_path,

                        progress_box

                    )
                )


            # =================================================
            # PROCESS FAILED
            # =================================================

            if return_code != 0:

                st.error(
                    "❌ Person Re-ID processing failed."
                )


                if error:

                    st.code(

                        error,

                        language="text"

                    )


                elif output:

                    st.code(

                        output,

                        language="text"

                    )


            # =================================================
            # PROCESS SUCCESSFUL
            # =================================================

            else:

                st.success(
                    "✅ Person search completed."
                )


                # ------------------------------------------------
                # Show final processing output
                # ------------------------------------------------

                if output:

                    with st.expander(
                        "📋 View Re-ID Processing Log",
                        expanded=False
                    ):

                        st.code(

                            output,

                            language="text"

                        )


                # ------------------------------------------------
                # READ DATABASE RESULTS
                # ------------------------------------------------

                df = get_person_sightings(
                    saved_filename
                )


                # =================================================
                # PERSON NOT FOUND
                # =================================================

                if df.empty:

                    st.warning(
                        "❌ Person not found in available cameras."
                    )


                # =================================================
                # PERSON FOUND
                # =================================================

                else:

                    st.success(

                        f"✅ Person found in "
                        f"{df['camera_id'].nunique()} camera(s)."

                    )


                    # ------------------------------------------------
                    # BEST RESULT
                    # ------------------------------------------------

                    best_row = df.loc[

                        df[
                            "best_similarity"
                        ].idxmax()

                    ]


                    # ------------------------------------------------
                    # METRICS
                    # ------------------------------------------------

                    col1, col2, col3, col4 = (
                        st.columns(4)
                    )


                    with col1:

                        st.metric(

                            "Camera Sightings",

                            len(df)

                        )


                    with col2:

                        st.metric(

                            "Cameras",

                            df[
                                "camera_id"
                            ].nunique()

                        )


                    with col3:

                        st.metric(

                            "Best Similarity",

                            f"{best_row['best_similarity']:.3f}"

                        )


                    with col4:

                        st.metric(

                            "Best Camera",

                            best_row[
                                "camera_id"
                            ]

                        )


                    st.divider()


                    # ------------------------------------------------
                    # BEST MATCH
                    # ------------------------------------------------

                    st.subheader(
                        "📍 Best Match"
                    )


                    st.info(

                        f"""
                        **Camera:** {best_row['camera_id']}

                        **City:** {best_row['city']}

                        **Road:** {best_row['road']}

                        **Video:** {best_row['video_source']}

                        **First Seen:** {best_row['first_seen_timestamp']}

                        **Last Seen:** {best_row['last_seen_timestamp']}

                        **Similarity:** {best_row['best_similarity']:.3f}
                        """

                    )


                    # ------------------------------------------------
                    # ALL SIGHTINGS
                    # ------------------------------------------------

                    st.subheader(
                        "📋 All Person Sightings"
                    )


                    st.dataframe(

                        df,

                        use_container_width=True,

                        hide_index=True

                    )


# ============================================================
# UPLOAD VIDEO
#
# Saves the uploaded file directly into assets\Camera_XX\ - the
# exact same location videos are normally copied into manually.
# Nothing downstream changes: Person Search and any other tool
# that scans Camera_* folders will pick the new video up on its
# next run automatically.
# ============================================================

elif mode == "📹 Upload Video":

    st.header(
        "📹 Upload Camera Video"
    )

    html_block(
        """
        <p class="section-lead">
        Upload a video for a specific camera. It is saved directly into
        that camera's folder under <code>assets/</code> - the same place
        videos are normally copied into by hand. Nothing else about the
        pipeline changes; existing tools (Person Search, vehicle
        detection) will scan it the next time they run.
        </p>
        """
    )

    # --------------------------------------------------------
    # DISCOVER EXISTING CAMERA FOLDERS
    # --------------------------------------------------------

    existing_cameras = []

    if os.path.exists(ASSETS_DIR):

        for item in sorted(os.listdir(ASSETS_DIR)):

            item_path = os.path.join(ASSETS_DIR, item)

            if os.path.isdir(item_path) and item.startswith("Camera_"):

                existing_cameras.append(item)

    NEW_CAMERA_OPTION = "➕ Create a new camera folder"

    camera_choice = st.selectbox(

        "Select Camera",

        existing_cameras + [NEW_CAMERA_OPTION],

        key="upload_video_camera_choice"

    )

    if camera_choice == NEW_CAMERA_OPTION:

        new_camera_name = st.text_input(

            "New camera folder name",

            placeholder="Example: Camera_04",

            key="upload_video_new_camera_name"

        )

        target_camera = new_camera_name.strip()

    else:

        target_camera = camera_choice

    st.divider()

    # --------------------------------------------------------
    # FILE UPLOADER
    # --------------------------------------------------------

    uploaded_video = st.file_uploader(

        "Choose a video file",

        type=["mp4", "avi", "mov", "mkv"],

        key="upload_video_file"

    )

    if uploaded_video is not None:

        size_mb = len(uploaded_video.getbuffer()) / (1024 * 1024)

        st.caption(
            f"Selected: {uploaded_video.name}  ({size_mb:.1f} MB)"
        )

    upload_button = st.button(

        "⬆️ UPLOAD VIDEO",

        type="primary",

        key="upload_video_button"

    )

    if upload_button:

        if not target_camera:

            st.warning(
                "Please select a camera or enter a name for the new "
                "camera folder."
            )

        elif uploaded_video is None:

            st.warning(
                "Please choose a video file first."
            )

        else:

            camera_dir = os.path.join(ASSETS_DIR, target_camera)

            os.makedirs(camera_dir, exist_ok=True)

            save_path = os.path.join(camera_dir, uploaded_video.name)

            already_existed = os.path.exists(save_path)

            with open(save_path, "wb") as f:

                f.write(uploaded_video.getbuffer())

            if already_existed:

                st.warning(
                    f"⚠️ A file named '{uploaded_video.name}' already "
                    f"existed in {target_camera} and has been overwritten."
                )

            st.success(
                f"✅ Video saved to: assets/{target_camera}/{uploaded_video.name}"
            )

            st.info(
                "This video will be included automatically the next "
                "time a search or detection is run - no further action "
                "is needed."
            )

    # --------------------------------------------------------
    # CURRENT VIDEOS PER CAMERA
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "📂 Videos currently in each camera folder"
    )

    if not existing_cameras:

        st.caption(
            "No camera folders exist yet. Upload a video above to "
            "create the first one."
        )

    else:

        for camera_name in existing_cameras:

            camera_dir = os.path.join(ASSETS_DIR, camera_name)

            videos_in_camera = [
                f for f in sorted(os.listdir(camera_dir))
                if f.lower().endswith(VIDEO_EXTENSIONS)
            ]

            with st.expander(
                f"{camera_name}  ({len(videos_in_camera)} video(s))"
            ):

                if videos_in_camera:

                    for video_name in videos_in_camera:

                        video_path = os.path.join(camera_dir, video_name)

                        video_size_mb = (
                            os.path.getsize(video_path) / (1024 * 1024)
                        )

                        st.write(
                            f"- {video_name}  ({video_size_mb:.1f} MB)"
                        )

                else:

                    st.caption(
                        "No videos in this camera folder yet."
                    )




    st.header(
        "🚑 Accident Emergency Response"
    )


    html_block(
        """
        <p class="section-lead">
        Provide the accident location by GPS or by typing an address. The
        system will find its coordinates, nearest hospital, available
        driving routes, fastest route and shortest route.
        </p>
        """
    )


    # --------------------------------------------------------
    # LOCATION METHOD CHOICE  (tabs - clear, high-contrast,
    # and the active tab is highlighted automatically)
    # --------------------------------------------------------

    tab_gps, tab_manual = st.tabs(
        [
            "📍  Use My Current Location (GPS)",
            "⌨️  Enter Location Manually"
        ]
    )


    # ==========================================================
    # GPS TAB
    # ==========================================================

    with tab_gps:

        st.caption(
            "Click the compass icon below and allow location "
            "access when your browser asks for permission."
        )

        gps_location = streamlit_geolocation()

        find_button_gps = st.button(

            "🚨 FIND NEAREST HOSPITAL & ROUTES",

            type="primary",

            key="find_hospital_routes_gps"

        )

        if find_button_gps:

            if not gps_location or gps_location.get("latitude") is None:

                st.warning(

                    "Location not captured yet. Click the compass "
                    "icon above, allow permission, then press this "
                    "button again."

                )

            else:

                latitude = gps_location["latitude"]

                longitude = gps_location["longitude"]

                with st.spinner(

                    "Finding your address, nearest hospital and routes..."

                ):

                    display_name = reverse_geocode(

                        latitude,

                        longitude

                    )

                    location = {

                        "latitude":
                            latitude,

                        "longitude":
                            longitude,

                        "display_name":
                            display_name

                    }

                    result = calculate_accident_response_from_location(

                        location,

                        search_label="📍 Current GPS Location"

                    )

                if result.get("error"):

                    st.error(
                        result["error"]
                    )

                    st.session_state.accident_result = None

                else:

                    st.session_state.accident_result = result


    # ==========================================================
    # MANUAL TAB
    # ==========================================================

    with tab_manual:

        location_name = st.text_input(

            "📍 Enter Location",

            placeholder="Example: Liberty Market Lahore",

            key="accident_location"

        )

        st.caption(

            "Examples: Mall Road Lahore, Gulberg Lahore, "
            "DHA Phase 5 Lahore, Liberty Market Lahore"

        )

        search_hospitals = st.button(

            "🚨 FIND NEAREST HOSPITAL & ROUTES",

            type="primary",

            key="find_hospital_routes"

        )

        if search_hospitals:

            if not location_name.strip():

                st.warning(
                    "Please enter a location."
                )

            else:

                with st.spinner(

                    "Finding location, nearest hospital and routes..."

                ):

                    result = calculate_accident_response(
                        location_name
                    )

                if result.get("error"):

                    st.error(
                        result["error"]
                    )

                    st.session_state.accident_result = None

                else:

                    st.session_state.accident_result = result


    # --------------------------------------------------------
    # DISPLAY SAVED RESULT
    # --------------------------------------------------------

    result = (
        st.session_state.accident_result
    )


    if result is not None:

        location = result["location"]

        search_label = result.get(
            "search_label",
            location["display_name"]
        )

        hospitals = result["hospitals"]

        nearest = result["nearest_hospital"]

        routes = result["routes"]

        fastest = result["fastest"]

        shortest = result["shortest"]


        # ----------------------------------------------------
        # LOCATION RESULT
        # ----------------------------------------------------

        st.success(
            "✅ Location found successfully."
        )


        st.info(

            f"""
            **Searched Location:** {search_label}

            **Matched Location:**
            {location['display_name']}

            **Latitude:** {location['latitude']}

            **Longitude:** {location['longitude']}
            """

        )


        # ----------------------------------------------------
        # NEAREST HOSPITAL
        # ----------------------------------------------------

        st.subheader(
            "🏥 Nearest Hospital"
        )


        col1, col2, col3 = st.columns(3)


        with col1:

            st.metric(

                "Hospital",

                nearest["name"]

            )


        with col2:

            st.metric(

                "Straight-line Distance",

                f"{nearest['distance_km']} km"

            )


        with col3:

            st.metric(

                "Available Routes",

                len(routes)

            )


        st.info(

            f"""
            **Hospital:** {nearest['name']}

            **Address:** {nearest.get('address', 'N/A')}

            **Distance:** {nearest['distance_km']} km
            """

        )


        # ----------------------------------------------------
        # ALL NEARBY HOSPITALS
        # ----------------------------------------------------

        st.subheader(
            "🏥 Nearby Hospitals"
        )


        hospital_data = []


        for hospital in hospitals:

            hospital_data.append(

                {

                    "Hospital":
                        hospital.get("name"),

                    "Distance (km)":
                        hospital.get("distance_km"),

                    "Address":
                        hospital.get(
                            "address",
                            "N/A"
                        )

                }

            )


        st.dataframe(

            pd.DataFrame(
                hospital_data
            ),

            use_container_width=True,

            hide_index=True

        )


        # ----------------------------------------------------
        # ROUTES
        # ----------------------------------------------------

        st.subheader(
            "🛣️ Available Routes to Nearest Hospital"
        )


        route_data = []


        for index, route in enumerate(

            routes,

            start=1

        ):

            route_data.append(

                {

                    "Route":
                        route.get(
                            "route_number",
                            index
                        ),

                    "Distance (km)":
                        route.get(
                            "distance_km"
                        ),

                    "Estimated Time (min)":
                        route.get(
                            "duration_minutes"
                        )

                }

            )


        st.dataframe(

            pd.DataFrame(
                route_data
            ),

            use_container_width=True,

            hide_index=True

        )


        # ----------------------------------------------------
        # FASTEST / SHORTEST
        # ----------------------------------------------------

        st.subheader(
            "🚨 Recommended Routes"
        )


        col1, col2 = st.columns(2)


        with col1:

            st.success(

                f"""
                ⚡ **FASTEST ROUTE**

                Route:
                {fastest.get('route_number', 'N/A')}

                Distance:
                {fastest['distance_km']} km

                Estimated Time:
                {fastest['duration_minutes']} minutes
                """

            )


        with col2:

            st.info(

                f"""
                📏 **SHORTEST ROUTE**

                Route:
                {shortest.get('route_number', 'N/A')}

                Distance:
                {shortest['distance_km']} km

                Estimated Time:
                {shortest['duration_minutes']} minutes
                """

            )


        # ----------------------------------------------------
        # EMERGENCY SUMMARY
        # ----------------------------------------------------

        st.subheader(
            "🚑 Emergency Response Summary"
        )


        st.warning(

            f"""
            🚨 **Emergency Location**

            {location['display_name']}

            🏥 **Nearest Hospital**

            {nearest['name']}

            📍 **Hospital Distance**

            {nearest['distance_km']} km

            ⚡ **Fastest Route**

            Route {fastest.get('route_number', 'N/A')} —
            {fastest['distance_km']} km —
            {fastest['duration_minutes']} minutes

            📏 **Shortest Route**

            Route {shortest.get('route_number', 'N/A')} —
            {shortest['distance_km']} km —
            {shortest['duration_minutes']} minutes
            """

        )


        # ----------------------------------------------------
        # MAP
        # ----------------------------------------------------

        st.subheader(
            "🗺️ Emergency Route Map"
        )


        route_map = result["map"]


        st_folium(

            route_map,

            width="100%",

            height=650,

            key="stable_emergency_route_map",

            returned_objects=[]

        )


# ============================================================
# FOOTER
# ============================================================

st.divider()


html_block(
    """
    <p class="app-footer">
    AI Accident Response System &nbsp;·&nbsp;
    YOLO + EasyOCR + Face Recognition Re-ID + SQLite + Hospital Routing + Folium + OpenStreetMap
    </p>
    """
)