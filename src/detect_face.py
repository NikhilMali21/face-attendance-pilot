import cv2
import os

# -----------------------------------
# Configuration
# -----------------------------------

MODEL_PATH = "models/face_detection_yunet_2023mar.onnx"

# Check model exists
if not os.path.exists(MODEL_PATH):
    print("ERROR: YuNet model not found!")
    print(f"Expected location: {MODEL_PATH}")
    exit()

# -----------------------------------
# Create YuNet Face Detector
# -----------------------------------

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

print("===================================")
print(" Face Detection Started")
print("===================================")
print("Press Q to quit.")

while True:

    # Read webcam frame
    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read frame.")
        break

    # Get frame dimensions
    height, width = frame.shape[:2]

    # Tell detector current image size
    detector.setInputSize((width, height))

    # Detect faces
    _, faces = detector.detect(frame)

    # Number of faces
    face_count = 0 if faces is None else len(faces)

    # Draw detections
    if faces is not None:

        for face in faces:

            # First 4 values = bounding box
            x, y, w, h = face[:4]

            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)

            # Draw rectangle
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # Label
            cv2.putText(
                frame,
                "Face Detected",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    # Display face count
    cv2.putText(
        frame,
        f"Faces: {face_count}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2
    )

    # Show webcam
    cv2.imshow(
        "Face Attendance - Detection",
        frame
    )

    # Quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# -----------------------------------
# Cleanup
# -----------------------------------

cap.release()
cv2.destroyAllWindows()

print("Face detection stopped.")