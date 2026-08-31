import os
import pickle
import numpy as np
import faiss

# ==============================
# Database Folder
# ==============================

folder = "vehicle_database"

features = []
filenames = []

print("=" * 50)
print("Loading Feature Files...")
print("=" * 50)

for file in os.listdir(folder):

    if file.endswith(".npy"):

        path = os.path.join(folder, file)

        feature = np.load(path).astype("float32")

        # Normalize vector
        feature /= np.linalg.norm(feature)

        features.append(feature)

        filenames.append(file.replace(".npy", ".jpg"))

        print(f"Loaded: {file}")

print("\nTotal Feature Vectors:", len(features))

# ==============================
# Convert to NumPy Matrix
# ==============================

features = np.vstack(features).astype("float32")

print("\nVector Shape:", features.shape)

# ==============================
# Build FAISS Index
# ==============================

dimension = features.shape[1]

index = faiss.IndexFlatIP(dimension)

index.add(features)

print("\nFAISS Index Created Successfully!")

print("Total Indexed Images:", index.ntotal)

# ==============================
# Save Index
# ==============================

os.makedirs("faiss_db", exist_ok=True)

faiss.write_index(index, "faiss_db/index.faiss")

with open("faiss_db/filenames.pkl", "wb") as f:
    pickle.dump(filenames, f)

print("\nIndex Saved Successfully!")

print("Location: faiss_db/index.faiss")