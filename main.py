import cv2
import mediapipe as mp
import numpy as np
import time
import math


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

def calculate_EAR(eye_points):
    p1 = eye_points[0]
    p2 = eye_points[1]
    p3 = eye_points[2]
    p4 = eye_points[3]
    p5 = eye_points[4]
    p6 = eye_points[5]

    vertical1 = distance(p2 , p6)
    vertical2 = distance(p3 , p5)
    horizontal = distance(p1 , p4)

    if horizontal == 0:
        return 0
    
    EAR = (vertical1 + vertical2) / (2 * horizontal)
    return EAR

def calculate_MAR(mouth_points):
    p1 = mouth_points[0]
    p2 = mouth_points[1]
    p3 = mouth_points[2]
    p4 = mouth_points[3]
    p5 = mouth_points[4]
    p6 = mouth_points[5]

    vertical1 = distance(p2 , p6)
    vertical2 = distance(p3 , p5)
    horizontal = distance(p1 , p4)

    if horizontal == 0:
        return 0
    
    MAR = (vertical1 + vertical2) / (2 * horizontal)
    return MAR

##Threshold-timer
EAR_threshold = 0.15 
closed_eye_time = None
eye_time = 1.5

MAR_threshold = 0.55
yawn_start_time = None
yawn_time = 1

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

        left_ear = calculate_EAR(left_points)
        right_ear = calculate_EAR(right_points)

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
        else:
            drowsy_by_eye = False

        #yawn-detection
        mouth = [61, 13, 14, 291, 78, 308]

        mouth_points = []
        for index in mouth:

            landmark = face_landmarks[index]

            x = int(landmark.x * w)
            y = int(landmark.y * h)

            mouth_points.append((x, y))
        
        mar = calculate_MAR(mouth_points)

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
        else:
            yawn_detect = False

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

            elif pitch > 10:
                vertical_status = "HEAD UP"

            else:
                vertical_status = "HEAD NORMAL"

            if yaw > 20:
                horizontal_status = "RIGHT"

            elif yaw < -20:
                horizontal_status = "LEFT"

            else:
                horizontal_status = "CENTER"

        else:
            pitch = 0
            yaw = 0
            roll = 0

            vertical_status = "UNKNOWN"
            horizontal_status = "UNKNOWN"

        #display
        cv2.putText(
            frame,
            f"EAR L:{left_ear:.2f} R:{right_ear:.2f}",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Eyes: {eye_status}",
            (30, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Eye closed: {closed_duration:.1f}s",
            (30, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"MAR: {mar:.2f}",
            (30, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Yawn: {'YES' if yawn_detect else 'NO'}",
            (30, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Pitch: {pitch:.1f}",
            (30, 190),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Yaw: {yaw:.1f}",
            (30, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Vertical: {vertical_status}",
            (30, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Horizontal: {horizontal_status}",
            (30, 280),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

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


