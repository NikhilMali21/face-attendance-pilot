import cv2
import os

# -----------------------------------
# Configuration
# -----------------------------------

STUDENT_NAME = "Nikhil"
SAVE_DIR = f"data/faces/{STUDENT_NAME}"

TOTAL_IMAGES = 10

os.makedirs(SAVE_DIR, exist_ok=True)

# -----------------------------------
# YuNet Face Detector
# -----------------------------------

MODEL_PATH = "models/face_detection_yunet_2023mar.onnx"

detector = cv2.FaceDetectorYN.create(
    MODEL_PATH,
    "",
    (320, 320),
    0.9,
    0.3,
    5000
)

# -----------------------------------
# Open Webcam
# -----------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("======================================")
print(" Face Dataset Capture")
print("======================================")
print("Student:", STUDENT_NAME)
print("Press SPACE to capture")
print("Press Q to quit")
print()

count = 0

while count < TOTAL_IMAGES:

    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read webcam frame.")
        break

    # -----------------------------------
    # Detect face
    # -----------------------------------

    height, width = frame.shape[:2]

    detector.setInputSize((width, height))

    _, faces = detector.detect(frame)

    face_count = 0 if faces is None else len(faces)

    # -----------------------------------
    # Draw face detection
    # -----------------------------------

    if faces is not None:

        for face in faces:

            x, y, w, h = face[:4]

            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

    # -----------------------------------
    # Display information
    # -----------------------------------

    cv2.putText(
        frame,
        f"Images: {count}/{TOTAL_IMAGES}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "SPACE = Capture | Q = Quit",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Faces: {face_count}",
        (20, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "Face Attendance - Capture",
        frame
    )

    # -----------------------------------
    # Keyboard controls
    # -----------------------------------

    key = cv2.waitKey(1) & 0xFF

    # SPACE
    if key == 32:

        # Capture only when exactly one face exists
        if faces is not None and len(faces) == 1:

            x, y, w, h = faces[0][:4]

            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)

            # Add padding around face
            padding = 20

            x1 = max(0, x - padding)
            y1 = max(0, y - padding)

            x2 = min(width, x + w + padding)
            y2 = min(height, y + h + padding)

            face_crop = frame[y1:y2, x1:x2]

            filename = f"{SAVE_DIR}/{count + 1}.jpg"

            cv2.imwrite(filename, face_crop)

            count += 1

            print(f"Captured {count}/{TOTAL_IMAGES}: {filename}")

        else:

            print("Please make sure exactly ONE face is visible.")

    # Q
    elif key == ord("q"):

        print("Capture stopped by user.")
        break


# -----------------------------------
# Cleanup
# -----------------------------------

cap.release()
cv2.destroyAllWindows()

print()
print("======================================")
print(" Capture Completed")
print("======================================")
print(f"Total images captured: {count}")
print(f"Saved in: {SAVE_DIR}")