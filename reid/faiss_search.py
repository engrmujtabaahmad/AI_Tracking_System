import os
import cv2
import faiss
import pickle
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torchvision import models

# ---------------------------------
# Load ResNet50 Feature Extractor
# ---------------------------------
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()

# ---------------------------------
# Image Transform
# ---------------------------------
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ---------------------------------
# Load FAISS Index
# ---------------------------------
index = faiss.read_index("faiss_db/index.faiss")

with open("faiss_db/filenames.pkl", "rb") as f:
    filenames = pickle.load(f)

# ---------------------------------
# Load Metadata
# ---------------------------------
metadata = pd.read_csv("database/vehicle_metadata.csv")

# ---------------------------------
# Load Query Image
# ---------------------------------
query_path = "queries/query.jpg"

image = cv2.imread(query_path)

if image is None:
    print("Query image not found!")
    exit()

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

tensor = transform(image)
tensor = tensor.unsqueeze(0)

# ---------------------------------
# Extract Feature Vector
# ---------------------------------
with torch.no_grad():
    feature = model(tensor)

feature = feature.squeeze().numpy().astype("float32")

# Normalize Feature Vector
feature = feature / np.linalg.norm(feature)

# ---------------------------------
# Search Top 5 Matches
# ---------------------------------
k = 5

distances, indices = index.search(
    np.expand_dims(feature, axis=0),
    k
)

print("=" * 60)
print("VEHICLE SEARCH RESULTS")
print("=" * 60)

for rank, (idx, score) in enumerate(zip(indices[0], distances[0]), start=1):

    image_name = filenames[idx]

    print(f"\nResult #{rank}")
    print("-" * 40)

    print(f"Image       : {image_name}")
    print(f"Similarity  : {score * 100:.2f}%")

    row = metadata[metadata["Image"] == image_name]

    if not row.empty:

        row = row.iloc[0]

        print(f"Camera ID   : {row['CameraID']}")
        print(f"City        : {row['Location']}")
        print(f"Road        : {row['Road']}")
        print(f"Timestamp   : {row['Timestamp']}")
        print(f"Frame       : {row['Frame']}")
        print(f"Vehicle Type: {row['VehicleType']}")

    else:

        print("Metadata Not Found!")

    print("-" * 40)

print("\nSearch Completed Successfully!")