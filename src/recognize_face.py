import cv2
import pickle
import numpy as np

from insightface.app import FaceAnalysis


# ==========================================
# Configuration
# ==========================================

EMBEDDING_FILE = "models/face_embeddings.pkl"

THRESHOLD = 0.40


# ==========================================
# Load saved embedding
# ==========================================

print("Loading saved face embedding...")

with open(EMBEDDING_FILE, "rb") as file:
    data = pickle.load(file)

student_name = data["name"]
reference_embedding = data["reference_embedding"]

print(f"Registered student: {student_name}")


# ==========================================
# Load InsightFace
# ==========================================

print("Loading InsightFace...")

app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)

print("InsightFace loaded!")
print()


# ==========================================
# Open webcam
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Could not open webcam.")
    exit()


print("==========================================")
print(" LIVE FACE RECOGNITION")
print("==========================================")
print("Press Q to quit.")


# ==========================================
# Cosine similarity function
# ==========================================

def cosine_similarity(a, b):

    return np.dot(a, b) / (
        np.linalg.norm(a) *
        np.linalg.norm(b)
    )


# ==========================================
# Live recognition loop
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
    # Process detected faces
    # --------------------------------------

    if len(faces) > 0:

        for face in faces:

            # Face bounding box
            x1, y1, x2, y2 = face.bbox.astype(int)

            # Get embedding
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
            # Determine identity
            # ----------------------------------

            if similarity >= THRESHOLD:

                name = student_name
                status = "MATCH"

                box_color = (0, 255, 0)

            else:

                name = "Unknown"
                status = "NO MATCH"

                box_color = (0, 0, 255)


            # ----------------------------------
            # Draw face box
            # ----------------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                box_color,
                2
            )


            # ----------------------------------
            # Display name
            # ----------------------------------

            cv2.putText(
                frame,
                f"Name: {name}",
                (x1, y1 - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                box_color,
                2
            )


            # ----------------------------------
            # Display similarity
            # ----------------------------------

            cv2.putText(
                frame,
                f"Similarity: {similarity:.2f}",
                (x1, y1 - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                box_color,
                2
            )


    # --------------------------------------
    # Display
    # --------------------------------------

    cv2.imshow(
        "Face Attendance - Recognition",
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
print("Recognition stopped.")