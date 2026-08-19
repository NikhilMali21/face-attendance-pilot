import os
import pickle
import cv2
import numpy as np

from insightface.app import FaceAnalysis


# ==========================================
# Configuration
# ==========================================

STUDENT_NAME = "Nikhil"

FACE_DIR = f"data/faces/{STUDENT_NAME}"

OUTPUT_DIR = "models"
OUTPUT_FILE = f"{OUTPUT_DIR}/face_embeddings.pkl"


# ==========================================
# Create output directory
# ==========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


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
    det_size=(320, 320)
)

print("InsightFace loaded successfully!")


# ==========================================
# Find images
# ==========================================

image_files = sorted(
    [
        file
        for file in os.listdir(FACE_DIR)
        if file.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]
)


print()
print("==========================================")
print(" FACE EMBEDDING GENERATION")
print("==========================================")
print(f"Student       : {STUDENT_NAME}")
print(f"Images found  : {len(image_files)}")
print()


# ==========================================
# Generate embeddings
# ==========================================

embeddings = []


for image_file in image_files:

    image_path = os.path.join(
        FACE_DIR,
        image_file
    )

    print(f"Processing: {image_file}")

    # Read image
    image = cv2.imread(image_path)

    if image is None:

        print("  ERROR: Could not read image.")
        continue

    # InsightFace detection + recognition
    faces = app.get(image)

    if len(faces) == 0:

        print("  WARNING: No face detected.")
        continue

    if len(faces) > 1:

        print("  WARNING: Multiple faces detected.")
        continue

    # Extract embedding
    embedding = faces[0].embedding

    # Normalize embedding
    embedding = embedding / np.linalg.norm(embedding)

    embeddings.append(embedding)

    print(
        f"  Face detected ✓ | "
        f"Embedding: {embedding.shape}"
    )


# ==========================================
# Validate
# ==========================================

if len(embeddings) == 0:

    print()
    print("ERROR: No embeddings generated.")
    print("Please check the face images.")
    exit()


# ==========================================
# Convert to NumPy array
# ==========================================

embeddings = np.array(embeddings)


# ==========================================
# Create reference embedding
# ==========================================

reference_embedding = np.mean(
    embeddings,
    axis=0
)

# Normalize reference embedding

reference_embedding = (
    reference_embedding /
    np.linalg.norm(reference_embedding)
)


# ==========================================
# Save
# ==========================================

data = {

    "student_id": "ST001",

    "name": STUDENT_NAME,

    "embeddings": embeddings,

    "reference_embedding": reference_embedding

}


with open(
    OUTPUT_FILE,
    "wb"
) as file:

    pickle.dump(
        data,
        file
    )


# ==========================================
# Final result
# ==========================================

print()
print("==========================================")
print(" EMBEDDINGS GENERATED SUCCESSFULLY")
print("==========================================")

print(
    f"Valid images : "
    f"{len(embeddings)}/{len(image_files)}"
)

print(
    f"Embeddings   : "
    f"{embeddings.shape}"
)

print(
    f"Reference    : "
    f"{reference_embedding.shape}"
)

print(
    f"Saved to     : "
    f"{OUTPUT_FILE}"
)

print("==========================================")