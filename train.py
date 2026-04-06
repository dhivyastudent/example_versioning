import os
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
import pickle

DATA_DIR = "samples"

def load_image(path):
    img = Image.open(path).resize((64, 64))
    return np.array(img).flatten()

X = []
y = []
labels = []

# Load dataset
for label in os.listdir(DATA_DIR):
    label_path = os.path.join(DATA_DIR, label)

    if not os.path.isdir(label_path):
        continue

    for file in os.listdir(label_path):
        file_path = os.path.join(label_path, file)

        try:
            X.append(load_image(file_path))
            y.append(label)
        except:
            print(f"Skipping {file_path}")

X = np.array(X)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained on classes:", set(y))
