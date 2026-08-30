import cv2
import mediapipe as mp
import numpy as np
import time
import math
import pyttsx3
import serial

##Arduino
arduino = serial.Serial("COM3", 9600, timeout=1)
time.sleep(2)

##Media-pipe-landmarker
BaseOptions = mp.tasks.BaseOptions
FaceLandMarker = mp.tasks.vision.FaceLandmarker #class
FaceLandMarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandMarkerOptions(
    base_options = BaseOptions(
        model_asset_path = "face_landmarker.task"
    ),
    running_mode = VisionRunningMode.VIDEO,
    num_faces = 1
)

face_LandMarker = FaceLandMarker.create_from_options( #model
    options
)

##Functions
def distance(points1 , points2):
    x1,y1 = points1
    x2,y2 = points2

    return math.sqrt(
        (x2 - x1) ** 2 + (y2 - y1) ** 2 
    )

def calculate_Aspect_Ratio(points):
    p1 = points[0]
    p2 = points[1]
    p3 = points[2]
    p4 = points[3]
    p5 = points[4]
    p6 = points[5]

    vertical1 = distance(p2 , p6)
    vertical2 = distance(p3 , p5)
    horizontal = distance(p1 , p4)

    if horizontal == 0:
        return 0
    
    aspectR = (vertical1 + vertical2) / (2 * horizontal)
    return aspectR

def speak(text):
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    engine.say(text)
    engine.runAndWait()
    engine.stop()

def buzzer():
    arduino.write(b"BUZZER\n")

##Threshold-timer
EAR_threshold = 0.15 
closed_eye_time = None
eye_time = 1.5

MAR_threshold = 0.55
yawn_start_time = None
yawn_time = 1

head_down_time = None
down_time = 2

last_time_event = None

last_voice_time = None
voice_threshold = 20

last_buzzer_time = None
buzzer_threshold = 20

##Event-score
yawn_score = 1
eye_score = 2
head_score = 2
score = 0

yawn_counted = False
eye_counted = False
head_counted = False

##Head-pose
landmark_ids = {
    "nose": 1,
    "chin": 152,
    "left_eye": 33,
    "right_eye": 263,
    "left_mouth": 61,
    "right_mouth": 291
}

model_points = np.array([
    (0.0, 0.0, 0.0),          # Nose
    (0.0, -330.0, -65.0),     # Chin
    (-225.0, 170.0, -135.0),  # Left eye
    (225.0, 170.0, -135.0),   # Right eye
    (-150.0, -150.0, -125.0), # Left mouth
    (150.0, -150.0, -125.0)   # Right mouth
], dtype=np.float64)

##Main-loop-camera
cap = cv2.VideoCapture(0)
start_time = time.time()

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    h, w, _ = frame.shape

    #convert-mediapipe_img
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    timestamp_ms = int(
        (time.time() - start_time) * 1000
    )

    results = face_LandMarker.detect_for_video(
        mp_image,
        timestamp_ms
    )

    #if-face-detected
    if results.face_landmarks:
        face_landmarks = results.face_landmarks[0]

        #eye-detection
        left_eye = [362, 385, 387, 263, 373, 380]
        right_eye = [33, 160, 158, 133, 153, 144]

        left_points = []
        for index in left_eye:
            landmark = face_landmarks[index]

            x = int(landmark.x * w)
            y = int(landmark.y * h)

            left_points.append((x, y))

        right_points = []
        for index in right_eye:

            landmark = face_landmarks[index]

            x = int(landmark.x * w)
            y = int(landmark.y * h)

            right_points.append((x, y))

        left_ear = calculate_Aspect_Ratio(left_points)
        right_ear = calculate_Aspect_Ratio(right_points)

        current_time = time.time()

        if left_ear < EAR_threshold and right_ear < EAR_threshold:
            eye_status = "Closed"
            if closed_eye_time is None:
                closed_eye_time = current_time
            closed_duration = current_time - closed_eye_time 
        else:
            eye_status = "Open"
            closed_eye_time = None
            closed_duration = 0

        if closed_duration >= eye_time:
            drowsy_by_eye = True
            
            if not eye_counted:
                score += eye_score
                eye_counted = True
                last_time_event = current_time
        else:
            drowsy_by_eye = False
            eye_counted = False

        #yawn-detection
        mouth = [61, 13, 14, 291, 78, 308]

        mouth_points = []
        for index in mouth:

            landmark = face_landmarks[index]

            x = int(landmark.x * w)
            y = int(landmark.y * h)

            mouth_points.append((x, y))
        
        mar = calculate_Aspect_Ratio(mouth_points)

        current_time = time.time()
        if mar >= MAR_threshold:
            if yawn_start_time is None:
                yawn_start_time = current_time
            yawn_duration = current_time - yawn_start_time 
        else:
            yawn_start_time = None
            yawn_duration = 0

        if yawn_duration >= yawn_time:
            yawn_detect = True

            if not yawn_counted:
                score += yawn_score
                yawn_counted = True
                last_time_event = current_time
        else:
            yawn_detect = False
            yawn_counted = False

        #head-pose
        image_points = []

        for name, index in landmark_ids.items():

            landmark = face_landmarks[index]

            x = landmark.x * w
            y = landmark.y * h

            image_points.append((x, y))

        image_points = np.array(
            image_points,
            dtype=np.float64
        )

        focal_length = w

        center = (
            w / 2,
            h / 2
        )

        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1))

        success, rotation_vector, translation_vector = cv2.solvePnP(
            model_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if success:

            rotation_matrix, _ = cv2.Rodrigues(
                rotation_vector
            )

            sy = np.sqrt(
                rotation_matrix[0, 0] ** 2 +
                rotation_matrix[1, 0] ** 2
            )

            singular = sy < 1e-6

            if not singular:

                pitch = np.arctan2(
                    rotation_matrix[2, 1],
                    rotation_matrix[2, 2]
                )

                yaw = np.arctan2(
                    -rotation_matrix[2, 0],
                    sy
                )

                roll = np.arctan2(
                    rotation_matrix[1, 0],
                    rotation_matrix[0, 0]
                )

            else:

                pitch = np.arctan2(
                    -rotation_matrix[1, 2],
                    rotation_matrix[1, 1]
                )

                yaw = np.arctan2(
                    -rotation_matrix[2, 0],
                    sy
                )

                roll = 0

            pitch = np.degrees(pitch)
            if pitch > 0:
                pitch = 180 - pitch
            else:
                pitch = -180 - pitch
            yaw = np.degrees(yaw)
            roll = np.degrees(roll)

            if pitch < -10:
                vertical_status = "HEAD DOWN"

            elif pitch > 6:
                vertical_status = "HEAD UP"

            else:
                vertical_status = "HEAD NORMAL"

            if vertical_status == "HEAD DOWN":
                if head_down_time is None:
                    head_down_time = current_time
                down_duration = current_time - head_down_time 
            else:
                head_down_time = None
                down_duration = 0

            if down_duration >= down_time:
                drowsy_by_head = True
                if not head_counted:
                    score += head_score
                    head_counted = True
                    last_time_event = current_time
            else:
                drowsy_by_head = False
                head_counted = False

        else:
            pitch = 0
            yaw = 0
            roll = 0

            vertical_status = "UNKNOWN"

        if last_time_event is not None:
            if current_time - last_time_event >= 60:
                score = 0
                last_time_event = None

        if score >= 9:
            level = 3
        elif score >= 6:
            level = 2
        elif score >= 3:
            level = 1
        else:
            level = 0

        #window
        if level == 1:
            cv2.putText(
                frame,
                f"Open the windows",
                (30, 70),
                cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                0.65,
                (0, 0, 255),
                2
            )

        #voice
        if level == 2:
            current_time = time.time()
            if last_voice_time is None or current_time - last_voice_time >= voice_threshold:
                speak("you are drowsy.get some rest!")
                last_voice_time = current_time

        #buzzer
        if level == 3:
            current_time = time.time()
            if last_buzzer_time is None or current_time - last_buzzer_time >= buzzer_threshold:
                buzzer()
                last_buzzer_time = current_time

        cv2.putText(
            frame,
            f"Score: {score}",
            (30, 50),
            cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
            0.65,
            (255, 0, 0),
            2
        )

        # cv2.putText(
        #     frame,
        #     f"Closed: {closed_duration:.2f}s",
        #     (30, 90),
        #     cv2.FONT_HERSHEY_SIMPLEX,
        #     1,
        #     (0, 255, 0),
        #     2
        # )

    else:
        cv2.putText(
            frame,
            "NO FACE DETECTED",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    cv2.imshow(
        "Drowsiness Detection",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()