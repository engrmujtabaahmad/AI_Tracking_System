import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

from torchvision import models

# -----------------------------
# Load Pretrained ResNet50
# -----------------------------
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# Remove classification layer
model = torch.nn.Sequential(*list(model.children())[:-1])

model.eval()

# -----------------------------
# Image Transform
# -----------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# -----------------------------
# Database Folder
# -----------------------------
folder = "vehicle_database"

print("="*50)
print("Extracting Features...")
print("="*50)

for filename in os.listdir(folder):

    path = os.path.join(folder, filename)

    image = cv2.imread(path)

    if image is None:
        continue

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    tensor = transform(image)

    tensor = tensor.unsqueeze(0)

    with torch.no_grad():
        feature = model(tensor)

    feature = feature.squeeze().numpy()

    save_name = filename.split(".")[0] + ".npy"

    np.save(
        os.path.join(folder, save_name),
        feature
    )

    print(f"Saved Features -> {save_name}")

print("\nCompleted Successfully!")