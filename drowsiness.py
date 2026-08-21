import cv2
import torch
import torch.nn as nn
from torchvision import models , transforms

##Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(
    model.last_channel,
    2
)

model.load_state_dict(torch.load("drowsiness_model.pth" , map_location=device))

model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

##Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
) 

classes = ["drowsy" , "notdrowsy"]

##Real-time detections
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret , frame = cap.read()
    if not ret:
        break

    ##Detect faces
    gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )

    # Draw rectangle around faces
    for (x, y, w, h) in faces:
        face = frame[
            y:y+h,
            x:x+w
        ]

        image = transform(face)
        image = image.unsqueeze(0)
        image = image.to(device)

        with torch.no_grad():
            outputs = model(image)

        probablities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(outputs , dim=1).item()

        label = classes[predicted_class]
        confidence = (probablities[0][predicted_class].item() * 100)

        ##Draw rectangle
        cv2.rectangle(
            frame,
            (x,y),
            (x+w , y+h),
            (0,255,0),
            2
        )
        # 10. Display Result
        # =========================

        text = f"{label}: {confidence:.1f}%"

        cv2.putText(
            frame,
            text,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(frame , label , (30,50) , cv2.FONT_HERSHEY_SIMPLEX ,
                    1 , (0,255,0) , 2)

    cv2.imshow("drowsiness detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()