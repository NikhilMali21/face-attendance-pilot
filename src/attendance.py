import cv2
import pickle
import numpy as np
import csv
import os
from datetime import datetime

from insightface.app import FaceAnalysis


# ==========================================
# Configuration
# ==========================================

EMBEDDING_FILE = "models/face_embeddings.pkl"

ATTENDANCE_DIR = "attendance"

ATTENDANCE_FILE = "attendance/attendance.csv"

THRESHOLD = 0.40


# ==========================================
# Create attendance directory
# ==========================================

os.makedirs(ATTENDANCE_DIR, exist_ok=True)


# ==========================================
# Load saved face embedding
# ==========================================

print("Loading registered student...")

with open(EMBEDDING_FILE, "rb") as file:
    data = pickle.load(file)


student_id = data["student_id"]
student_name = data["name"]
reference_embedding = data["reference_embedding"]


print(f"Student ID : {student_id}")
print(f"Student    : {student_name}")


# ==========================================
# Create CSV if it doesn't exist
# ==========================================

if not os.path.exists(ATTENDANCE_FILE):

    with open(
        ATTENDANCE_FILE,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "Student_ID",
                "Name",
                "Date",
                "Time",
                "Status"
            ]
        )

    print("Attendance file created.")


# ==========================================
# Check whether attendance already exists
# ==========================================

def already_marked_today(student_id):

    today = datetime.now().strftime("%Y-%m-%d")

    with open(
        ATTENDANCE_FILE,
        "r",
        newline=""
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            if (
                row["Student_ID"] == student_id
                and row["Date"] == today
            ):

                return True

    return False


# ==========================================
# Mark attendance
# ==========================================

def mark_attendance(student_id, name):

    if already_marked_today(student_id):

        print(
            f"Attendance already marked today "
            f"for {name}."
        )

        return False


    now = datetime.now()

    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")


    with open(
        ATTENDANCE_FILE,
        "a",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                student_id,
                name,
                date,
                time,
                "Present"
            ]
        )


    print()
    print("==========================================")
    print(" ATTENDANCE MARKED")
    print("==========================================")
    print(f"Student : {name}")
    print(f"Date    : {date}")
    print(f"Time    : {time}")
    print("Status  : Present")
    print("==========================================")


    return True


# ==========================================
# Cosine similarity
# ==========================================

def cosine_similarity(a, b):

    return np.dot(a, b) / (
        np.linalg.norm(a) *
        np.linalg.norm(b)
    )


# ==========================================
# Load InsightFace
# ==========================================

print()
print("Loading InsightFace...")

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)

print("InsightFace loaded.")


# ==========================================
# Open webcam
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Could not open webcam.")
    exit()


print()
print("==========================================")
print(" AUTOMATIC ATTENDANCE SYSTEM")
print("==========================================")
print("Press Q to quit.")
print()


# ==========================================
# Attendance state
# ==========================================

attendance_marked = already_marked_today(
    student_id
)


# ==========================================
# Main loop
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:

        print("ERROR: Could not read webcam.")
        break


    # --------------------------------------
    # Detect faces
    # --------------------------------------

    faces = app.get(frame)


    # --------------------------------------
    # Process faces
    # --------------------------------------

    for face in faces:

        # Bounding box

        x1, y1, x2, y2 = face.bbox.astype(int)


        # Get face embedding

        embedding = face.embedding


        # Normalize

        embedding = (
            embedding /
            np.linalg.norm(embedding)
        )


        # Compare

        similarity = cosine_similarity(
            embedding,
            reference_embedding
        )


        # ----------------------------------
        # Recognition
        # ----------------------------------

        if similarity >= THRESHOLD:

            name = student_name

            color = (0, 255, 0)

            status = "MATCH"


            # --------------------------------
            # Mark attendance
            # --------------------------------

            if not attendance_marked:

                marked = mark_attendance(
                    student_id,
                    student_name
                )

                if marked:

                    attendance_marked = True


        else:

            name = "Unknown"

            color = (0, 0, 255)

            status = "UNKNOWN"


        # ----------------------------------
        # Draw face rectangle
        # ----------------------------------

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            2
        )


        # ----------------------------------
        # Display name
        # ----------------------------------

        cv2.putText(
            frame,
            f"Name: {name}",
            (x1, y1 - 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )


        # ----------------------------------
        # Display similarity
        # ----------------------------------

        cv2.putText(
            frame,
            f"Similarity: {similarity:.2f}",
            (x1, y1 - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )


        # ----------------------------------
        # Attendance status
        # ----------------------------------

        if name == student_name:

            if attendance_marked:

                attendance_text = "PRESENT"

            else:

                attendance_text = "DETECTED"

        else:

            attendance_text = "NOT RECOGNIZED"


        cv2.putText(
            frame,
            attendance_text,
            (x1, y2 + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2
        )


    # --------------------------------------
    # Window title information
    # --------------------------------------

    cv2.putText(
        frame,
        "Automatic Attendance",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2
    )


    # --------------------------------------
    # Show frame
    # --------------------------------------

    cv2.imshow(
        "Face Attendance - Automatic Attendance",
        frame
    )


    # --------------------------------------
    # Quit
    # --------------------------------------

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ==========================================
# Cleanup
# ==========================================

cap.release()

cv2.destroyAllWindows()


print()
print("Attendance system stopped.")