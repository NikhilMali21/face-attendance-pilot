import streamlit as st
import pandas as pd
import cv2
import os
import time

from src.face_engine import FaceRecognitionEngine
from src.attendance_service import AttendanceService


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Face Attendance",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CONFIGURATION
# ============================================================

ATTENDANCE_FILE = "attendance/attendance.csv"
EMBEDDING_FILE = "models/face_embeddings.pkl"

STUDENT_COUNT = 1


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 18px;
        margin-top: 0px;
        opacity: 0.7;
    }

    .status-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.25);
        margin-bottom: 15px;
    }

    .success-text {
        font-size: 24px;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🤖 AI Face Attendance System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Real-Time Face Recognition & Automated Attendance Pilot'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# LOAD SERVICES
# ============================================================

@st.cache_resource
def load_face_engine():

    return FaceRecognitionEngine(
        embedding_file=EMBEDDING_FILE,
        threshold=0.40
    )


@st.cache_resource
def load_attendance_service():

    return AttendanceService(
        attendance_file=ATTENDANCE_FILE
    )


face_engine = load_face_engine()

attendance_service = load_attendance_service()


# ============================================================
# LOAD ATTENDANCE DATA
# ============================================================

def load_attendance():

    if os.path.exists(ATTENDANCE_FILE):

        try:

            return pd.read_csv(
                ATTENDANCE_FILE
            )

        except Exception:

            return pd.DataFrame(
                columns=[
                    "Student_ID",
                    "Name",
                    "Date",
                    "Time",
                    "Status"
                ]
            )

    return pd.DataFrame(
        columns=[
            "Student_ID",
            "Name",
            "Date",
            "Time",
            "Status"
        ]
    )


# ============================================================
# CURRENT DATE
# ============================================================

today = time.strftime(
    "%Y-%m-%d"
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ System")

    st.success(
        "System Online"
    )

    st.write(
        "**Recognition:** InsightFace"
    )

    st.write(
        "**Embedding:** 512-D"
    )

    st.write(
        "**Backend:** ONNX Runtime"
    )

    st.write(
        "**Storage:** CSV"
    )

    st.divider()

    st.subheader(
        "👤 Registered Student"
    )

    st.write(
        "ST001 — Nikhil"
    )

    st.divider()

    st.caption(
        "AI Face Attendance Pilot"
    )


# ============================================================
# LOAD DATA
# ============================================================

df = load_attendance()


today_attendance = df[
    df["Date"] == today
]


present_count = len(
    today_attendance[
        today_attendance["Status"]
        == "Present"
    ]
)


absent_count = max(
    STUDENT_COUNT - present_count,
    0
)


attendance_percentage = 0

if STUDENT_COUNT > 0:

    attendance_percentage = (
        present_count /
        STUDENT_COUNT
    ) * 100


# ============================================================
# METRICS
# ============================================================

st.subheader(
    "📊 Today's Overview"
)


metric1, metric2, metric3, metric4 = st.columns(4)


with metric1:

    st.metric(
        "👨‍🎓 Students",
        STUDENT_COUNT
    )


with metric2:

    st.metric(
        "✅ Present",
        present_count
    )


with metric3:

    st.metric(
        "❌ Absent",
        absent_count
    )


with metric4:

    st.metric(
        "📈 Attendance",
        f"{attendance_percentage:.0f}%"
    )


st.divider()


# ============================================================
# LIVE ATTENDANCE
# ============================================================

st.subheader(
    "📷 Live Attendance"
)


start_camera = st.checkbox(
    "▶ Start Attendance Camera"
)


if start_camera:

    st.info(
        "Camera is active. Look directly at the camera."
    )

    camera_placeholder = st.empty()

    recognition_placeholder = st.empty()

    cap = cv2.VideoCapture(0)


    if not cap.isOpened():

        st.error(
            "❌ Webcam could not be opened."
        )

    else:

        stop_camera = st.button(
            "⏹ Stop Camera"
        )


        while not stop_camera:

            ret, frame = cap.read()


            if not ret:

                st.error(
                    "❌ Could not read webcam frame."
                )

                break


            # ---------------------------------------------
            # RECOGNITION
            # ---------------------------------------------

            results = face_engine.recognize(
                frame
            )


            # ---------------------------------------------
            # PROCESS RESULTS
            # ---------------------------------------------

            for result in results:

                if result["matched"]:

                    student_id = result[
                        "student_id"
                    ]

                    student_name = result[
                        "name"
                    ]

                    similarity = result[
                        "similarity"
                    ]


                    attendance_result = (
                        attendance_service.mark_attendance(
                            student_id,
                            student_name
                        )
                    )


                    if attendance_result[
                        "success"
                    ]:

                        recognition_placeholder.success(
                            f"✅ {student_name} — "
                            f"Attendance Marked | "
                            f"Similarity: "
                            f"{similarity:.2f}"
                        )

                    elif attendance_result[
                        "already_marked"
                    ]:

                        recognition_placeholder.info(
                            f"🟢 {student_name} — "
                            f"Already Present Today | "
                            f"Similarity: "
                            f"{similarity:.2f}"
                        )

                else:

                    recognition_placeholder.warning(
                        "⚠️ Unknown face detected."
                    )


            # ---------------------------------------------
            # DRAW FACE BOXES
            # ---------------------------------------------

            frame = face_engine.draw_results(
                frame,
                results
            )


            # ---------------------------------------------
            # BGR → RGB
            # ---------------------------------------------

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )


            # ---------------------------------------------
            # DISPLAY
            # ---------------------------------------------

            camera_placeholder.image(
                frame,
                channels="RGB",
                use_container_width=True
            )


        cap.release()

        cv2.destroyAllWindows()


else:

    st.info(
        "Camera is currently stopped."
    )


st.divider()


# ============================================================
# TODAY'S ATTENDANCE
# ============================================================

st.subheader(
    "📋 Today's Attendance"
)


df = load_attendance()


today_attendance = df[
    df["Date"] == today
]


if len(today_attendance) > 0:

    display_df = today_attendance[
        [
            "Student_ID",
            "Name",
            "Time",
            "Status"
        ]
    ]

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "No attendance recorded today."
    )


st.divider()


# ============================================================
# ATTENDANCE HISTORY
# ============================================================

st.subheader(
    "📈 Attendance History"
)


if len(df) > 0:

    history_df = df[
        [
            "Student_ID",
            "Name",
            "Date",
            "Time",
            "Status"
        ]
    ]

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info(
        "No attendance history available."
    )


st.divider()


# ============================================================
# MODEL INFORMATION
# ============================================================

st.subheader(
    "🧠 AI Model Information"
)


info1, info2, info3 = st.columns(3)


with info1:

    st.write(
        "**Face Recognition**"
    )

    st.write(
        "InsightFace / ArcFace"
    )


with info2:

    st.write(
        "**Face Embedding**"
    )

    st.write(
        "512-dimensional vector"
    )


with info3:

    st.write(
        "**Matching Method**"
    )

    st.write(
        "Cosine Similarity"
    )


st.divider()


# ============================================================
# FOOTER
# ============================================================

st.caption(
    "AI Face Attendance Pilot • "
    "Built with Python, OpenCV, InsightFace & Streamlit"
)