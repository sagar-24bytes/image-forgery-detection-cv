import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np
import os

from src.features import extract_features, preprocess

# -------------------------------
# Check model exists
# -------------------------------
if not os.path.exists("model.pkl"):
    print("❌ Model not found! Run train.py first.")
    exit()

# Load model & scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

def predict_image(path):
    img = cv2.imread(path)

    if img is None:
        print("❌ Image not found")
        return

    img_resized = preprocess(img)

    # Feature extraction
    feat = extract_features(img)

    # NORMALIZE (IMPORTANT)
    feat = scaler.transform([feat])

    pred = model.predict(feat)[0]

    label = "Tampered" if pred == 1 else "Real"

    # -------------------------------
    # FFT visualization
    # -------------------------------
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    fft = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = np.log(np.abs(fft) + 1)

    # -------------------------------
    # Edge detection
    # -------------------------------
    edges = cv2.Canny(gray, 100, 200)

    # -------------------------------
    # Plot
    # -------------------------------
    plt.figure(figsize=(10,4))

    plt.subplot(1,3,1)
    plt.imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
    plt.title("Original")

    plt.subplot(1,3,2)
    plt.imshow(magnitude, cmap='gray')
    plt.title("FFT")

    plt.subplot(1,3,3)
    plt.imshow(edges, cmap='gray')
    plt.title("Edges")

    plt.suptitle(f"Prediction: {label}", fontsize=14)
    plt.show()


#  test image path
predict_image(r"B:\image-forgery-detection-cv\data\fake\copymove\00048.png")