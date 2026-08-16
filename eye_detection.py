import cv2
import mediapipe as mp
import time
import math

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

cap = cv2.VideoCapture(0)

start_time = time.time()

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

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

    if results.face_landmarks:
        for face_landmarks in results.face_landmarks:
            h, w, _ = frame.shape
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

            ear = (left_ear + right_ear) / 2

            cv2.putText(
                frame,
                f"EAR: {ear:.2f}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

    cv2.imshow(
        "Eye Detection",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()