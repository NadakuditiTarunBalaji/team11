"""import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import time
import threading
from playsound import playsound

# Load the trained model
model = load_model("drowsiness_detection_model1.keras")

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Landmark indices for eyes and mouth
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
MOUTH_IDX = [61, 81, 78, 13, 312, 308, 291, 402, 318, 324, 14, 88]

# Function to calculate Eye Aspect Ratio (EAR)
def EAR(eye_landmarks):
    point1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    point2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    distance = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    ear_aspect_ratio = (point1 + point2) / (2.0 * distance)
    return ear_aspect_ratio

# Function to calculate Mouth Aspect Ratio (MAR)
def MAR(mouth_landmarks):
    point = np.linalg.norm(mouth_landmarks[0] - mouth_landmarks[6])
    point1 = np.linalg.norm(mouth_landmarks[2] - mouth_landmarks[10])
    point2 = np.linalg.norm(mouth_landmarks[4] - mouth_landmarks[8])
    Ypoint = (point1 + point2) / 2.0
    mouth_aspect_ratio = Ypoint / point
    return mouth_aspect_ratio

# Function to play beep sound
def play_beep():
    playsound("beeep.mp3")  # Replace with the path to your beep sound file

# Open webcam and predict drowsiness
def predict_drowsiness():
    webcamera = cv2.VideoCapture(0)
    
    print("Press 'q' to quit.")
    
    drowsy_start_time = None  # Track when drowsy state starts
    
    while True:
        ret, frame = webcamera.read()
        frame = cv2.flip(frame, 1)  # Flip the frame horizontally for mirror view
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb_frame)

        if result.multi_face_landmarks:
            for face_landmarks in result.multi_face_landmarks:
                landmarks = np.array([[p.x, p.y] for p in face_landmarks.landmark])

                # Extract left and right eye landmarks
                left_eye = landmarks[LEFT_EYE_IDX]
                right_eye = landmarks[RIGHT_EYE_IDX]
                
                # Calculate EAR for eyes
                leftEAR = EAR(left_eye)
                rightEAR = EAR(right_eye)
                ear = (leftEAR + rightEAR) / 2.0
                
                # Extract mouth landmarks and calculate MAR
                mouth = landmarks[MOUTH_IDX]
                mouEAR = MAR(mouth)

                # Prepare input for the model
                features = np.array([[ear, mouEAR]])  # Model expects 2D input
                prediction = model.predict(features)
                label = "Drowsy" if prediction[0][0] > 0.5 else "Not Drowsy"

                # Display the prediction and features
                cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, f"MAR: {mouEAR:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, f"Prediction: {label}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Handle drowsiness state persistence
                if label == "Drowsy":
                    if drowsy_start_time is None:
                        drowsy_start_time = time.time()
                    elif time.time() - drowsy_start_time > 0.5:  # If drowsy for >0.5 seconds
                        threading.Thread(target=play_beep).start()
                else:
                    drowsy_start_time = None  # Reset if not drowsy

        # Check for key press to quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        cv2.imshow("Drowsiness Detection", frame)

    webcamera.release()
    cv2.destroyAllWindows()

# Run the prediction function
predict_drowsiness()
"""