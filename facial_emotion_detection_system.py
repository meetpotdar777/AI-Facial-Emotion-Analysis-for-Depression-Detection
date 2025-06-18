import cv2
import numpy as np
import os

# --- Configuration Paths ---
# IMPORTANT: Update FACE_CASCADE_PATH if cv2.data.haarcascades doesn't work.
# This should point to your 'haarcascade_frontalface_default.xml' file.
# Example: FACE_CASCADE_PATH = 'C:/Users/Administrator/OneDrive/Desktop/Python/AI Facial Emotion Analysis for Depression Detection/facial_emotion_env/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml'
FACE_CASCADE_PATH = 'C:/Users/Administrator/OneDrive/Desktop/Python/AI Facial Emotion Analysis for Depression Detection/facial_emotion_env/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml'

# --- Load Face Detection Model ---
try:
    face_classifier = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    if face_classifier.empty():
        raise IOError(f"Could not load face cascade from: {FACE_CASCADE_PATH}")
except IOError as e:
    print(f"Error loading face cascade: {e}")
    print("Please ensure 'haarcascade_frontalface_default.xml' is correctly located and the FACE_CASCADE_PATH is set.")
    print("Common location for cascade files: facial_emotion_env/Lib/site-packages/cv2/data/")
    exit() # Exit if the essential face cascade cannot be loaded

# --- Improved Rule-Based Emotion and Age Simulation (Still NOT Deep Learning) ---
# These functions will provide more noticeable changes in output
# based on simple visual cues, demonstrating the "workflow"
# without requiring actual AI models. They are for conceptual demo ONLY.

def simulate_emotion(face_roi):
    """
    Simulates emotion based on the area of the detected face.
    This provides a visible change in output as the user moves closer/further.
    """
    h, w = face_roi.shape[:2]
    face_area = h * w

    if face_area > 40000: # Larger face, closer to camera
        return "Emotion: High Expressiveness (Simulated)"
    elif face_area > 15000: # Medium face
        return "Emotion: Normal Expressiveness (Simulated)"
    else: # Smaller face, further from camera
        return "Emotion: Low Expressiveness (Simulated)"

def simulate_age(face_roi):
    """
    Simulates age category based on the height of the detected face.
    This is a very rough visual heuristic, NOT accurate age identification.
    """
    h, w = face_roi.shape[:2]

    # These thresholds are arbitrary and depend heavily on webcam distance
    # and face size. They are merely for demonstration of dynamic output.
    if h > 200: # Very large face on screen
        return "Age: Adult (Simulated)"
    elif h > 100: # Medium face on screen
        return "Age: Teen/Adult (Simulated)"
    else: # Smaller face on screen
        return "Age: Child/Youth (Simulated)"


# --- Main Webcam Loop ---

def run_face_detection_and_simulated_analysis():
    cap = cv2.VideoCapture(0) # 0 for default webcam

    if not cap.isOpened():
        print("Error: Could not open webcam. Make sure your webcam is connected and not in use.")
        return

    print("Webcam opened successfully. Detecting faces and simulating emotion/age. Press 'q' to quit.")
    print("Try moving closer/further from the camera to see simulated changes.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting...")
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_classifier.detectMultiScale(
            gray_frame,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2) # Green rectangle around face

            # Extract face region of interest (ROI) for analysis
            face_roi = frame[y:y+h, x:x+w]

            # --- Simulated Emotion Classification ---
            emotion_text = simulate_emotion(face_roi)
            # Display emotion text just above the face rectangle
            cv2.putText(frame, emotion_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # --- Simulated Age Identification ---
            age_text = simulate_age(face_roi)
            # Display age text just below the face rectangle
            cv2.putText(frame, age_text, (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2) # Blue color for age


        cv2.imshow('AI Facial Emotion & Age Analysis (Simulated)', frame)

        # Press 'q' to quit the application
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release webcam and destroy all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    print("Application closed. Remember, for mental health concerns, consult a professional.")

if __name__ == "__main__":
    run_face_detection_and_simulated_analysis()