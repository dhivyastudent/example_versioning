import numpy as np
from PIL import Image
import pickle

def load_image(path):
    img = Image.open(path).resize((64, 64))
    return np.array(img).flatten()

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Test image (change this if needed)
img_path = "samples/dog/dog1.jpg"

X_test = np.array([load_image(img_path)])

prediction = model.predict(X_test)[0]

# Save output
with open("prediction.txt", "w") as f:
    f.write(f"Prediction for {img_path}: {prediction}")

print("Prediction:", prediction)
