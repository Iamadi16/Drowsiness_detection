from ultralytics import YOLO
import torch 
from matplotlib import pyplot as plt
import numpy as np 
import cv2

##Load model
model = YOLO("yolov8n.pt")

##Make detections
img = '3.jpg'
results = model(img)
print(results)

annotated_img = results[0].plot()
plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)) #colors are different in cv2 and matplot
plt.show()
