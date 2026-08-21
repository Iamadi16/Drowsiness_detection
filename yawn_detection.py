import cv2
import mediapipe as mp
import time
import math

BaseOptions = mp.tasks.BaseOptions
FaceLandMarker = mp.tasks.vision.FaceLandmarker
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

MAR_threshold = 0.55
yawn_time = 1
yawn_start_time = None

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

            if yawn_detect:
                yawn_status = "YAWN"
            else:
                yawn_status = "NO YAWN"

            cv2.putText(
                frame,
                f"MAR: {mar:.2f}{yawn_status}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            cv2.putText(
                frame,
                f"Yawn time: {yawn_duration:.2f}s",
                (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )
     
    cv2.imshow(
        "yawn Detection",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()