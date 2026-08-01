from ultralytics import YOLO
import cv2

##Load model
model = YOLO("yolov8n.pt")

# annotated_img = results[0].plot()
# plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)) #colors are different in cv2 and matplot
# plt.show()

#Real-time detections
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret , frame = cap.read()
    if not ret:
        break

    ##Make detections
    results = model(frame , verbose = False) #verbos->delete each terminal info
    annotated_frame = results[0].plot()
    cv2.imshow("YOLO", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()