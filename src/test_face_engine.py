import cv2

from face_engine import FaceRecognitionEngine


print("==========================================")
print(" TESTING FACE RECOGNITION ENGINE")
print("==========================================")


# Load engine

engine = FaceRecognitionEngine()

print("Face engine loaded successfully.")


# Open webcam

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Webcam could not be opened.")
    exit()


print("Webcam started.")
print("Press Q to quit.")


while True:

    ret, frame = cap.read()

    if not ret:

        print("Could not read frame.")
        break


    # Recognize faces

    results = engine.recognize(
        frame
    )


    # Draw results

    frame = engine.draw_results(
        frame,
        results
    )


    # Display

    cv2.imshow(
        "Face Engine Test",
        frame
    )


    # Quit

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


cap.release()

cv2.destroyAllWindows()

print("Test completed.")