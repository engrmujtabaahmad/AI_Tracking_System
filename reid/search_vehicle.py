import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import models

# ==========================================
# Load ResNet50 Feature Extractor
# ==========================================

model = models.resnet50(
    weights=models.ResNet50_Weights.DEFAULT
)

# Remove last classification layer
model = torch.nn.Sequential(
    *list(model.children())[:-1]
)

model.eval()

# ==========================================
# Image Transform
# ==========================================

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ==========================================
# Load Query Image
# ==========================================

query_path = "queries/query.jpg"

query = cv2.imread(query_path)

if query is None:
    print("Query image not found!")
    exit()

query = cv2.cvtColor(query, cv2.COLOR_BGR2RGB)

tensor = transform(query)

tensor = tensor.unsqueeze(0)

# ==========================================
# Extract Query Features
# ==========================================

with torch.no_grad():
    query_feature = model(tensor)

query_feature = query_feature.squeeze().numpy()

# Normalize feature vector
query_feature = query_feature / np.linalg.norm(query_feature)

print("=" * 50)
print("Searching...")
print("=" * 50)

best_score = -1
best_match = ""

folder = "vehicle_database"

# ==========================================
# Compare With Every Stored Feature
# ==========================================

for file in os.listdir(folder):

    if not file.endswith(".npy"):
        continue

    feature = np.load(os.path.join(folder, file))

    feature = feature / np.linalg.norm(feature)

    similarity = np.dot(query_feature, feature)

    print(f"{file:25} Similarity: {similarity:.4f}")

    if similarity > best_score:
        best_score = similarity
        best_match = file

print("\n" + "=" * 50)
print("Best Match Found")
print("=" * 50)

print(f"Vehicle : {best_match.replace('.npy','.jpg')}")
print(f"Similarity : {best_score*100:.2f}%")