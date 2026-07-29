import torch 
from matplotlib import pyplot as plt
import numpy as np 
import cv2

##Load model
model = torch.hub.load('ultralytics/yolov5' , 'yolov5s')
