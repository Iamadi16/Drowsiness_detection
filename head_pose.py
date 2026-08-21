import cv2
import mediapipe as mp
import numpy as np
import time


BaseOptions = mp.tasks.BaseOptions
FaceLandMarker = mp.tasks.vision.FaceLandmarker
FaceLandMarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


# MediaPipe Face Landmarker
options = FaceLandMarkerOptions(
    base_options=BaseOptions(
        model_asset_path="face_landmarker.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1
)

face_landmarker = FaceLandMarker.create_from_options(options)


# Facial landmark indexes
landmark_ids = {
    "nose": 1,
    "chin": 152,
    "left_eye": 33,
    "right_eye": 263,
    "left_mouth": 61,
    "right_mouth": 291
}


# Generic 3D face model
model_points = np.array([
    (0.0, 0.0, 0.0),          # Nose
    (0.0, -330.0, -65.0),     # Chin
    (-225.0, 170.0, -135.0),  # Left eye
    (225.0, 170.0, -135.0),   # Right eye
    (-150.0, -150.0, -125.0), # Left mouth
    (150.0, -150.0, -125.0)   # Right mouth
], dtype=np.float64)


cap = cv2.VideoCapture(0)

start_time = time.time()


while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    h, w, _ = frame.shape

    # BGR → RGB
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

    results = face_landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )

    if results.face_landmarks:

        face_landmarks = results.face_landmarks[0]

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

        # Camera parameters
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


        # Solve PnP
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

            # Calculate Euler angles
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


            # Radians → Degrees
            pitch = np.degrees(pitch)
            if pitch > 0:
                pitch = 180 - pitch
            else:
                pitch = - 180 - pitch
            yaw = np.degrees(yaw)
            roll = np.degrees(roll)


            # Display angles
            cv2.putText(
                frame,
                f"Pitch: {pitch:.1f}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Yaw: {yaw:.1f}",
                (30, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Roll: {roll:.1f}",
                (30, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )


            # Head direction
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


            cv2.putText(
                frame,
                f"Vertical: {vertical_status}",
                (30, 165),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Horizontal: {horizontal_status}",
                (30, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )


    cv2.imshow(
        "Head Pose Detection",
        frame
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()