import cv2
import numpy as np
import torch
from ultralytics import YOLO

print("====================================")
print("Environment Test")
print("====================================")

print("OpenCV Version:", cv2.__version__)
print("NumPy Version:", np.__version__)
print("PyTorch Version:", torch.__version__)

print("CUDA Available:", torch.cuda.is_available())

print("Ultralytics imported successfully!")

print("====================================")
print("Everything is working!")
print("====================================")