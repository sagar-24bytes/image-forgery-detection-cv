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
    print("!! Model not found! Run train.py first.")
    exit()

# Load model & scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")


# -------------------------------
# Load mask function 
# -------------------------------
def load_mask(image_path):
    mask_path = image_path.replace("fake", "mask")

    
    if not os.path.exists(mask_path):
        mask_path = mask_path.replace(".jpg", ".png")

    mask = cv2.imread(mask_path, 0)

    if mask is None:
        return None

    mask = cv2.resize(mask, (128, 128))
    return mask


# -------------------------------
# Prediction function
# -------------------------------
def predict_image(path):
    img = cv2.imread(path)

    if img is None:
        print("!Image not found")
        return

    img_resized = preprocess(img)

    # Feature extraction
    feat = extract_features(img)

    # Normalize
    feat = scaler.transform([feat])

    pred = model.predict(feat)[0]
    label = "Tampered" if pred == 1 else "Real"

    # FFT
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    fft = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = np.log(np.abs(fft) + 1)

    # Edges
    edges = cv2.Canny(gray, 100, 200)

    # Load mask
    mask = load_mask(path)

    overlay = None
    if mask is not None:
        overlay = img_resized.copy()
        overlay[mask > 0] = [255, 0, 0]

    # -------------------------------
    # Plotting Section
    # -------------------------------
    plt.figure(figsize=(12,4))

    # Original
    plt.subplot(1,4,1)
    plt.imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
    plt.title("Original")

    # FFT
    plt.subplot(1,4,2)
    plt.imshow(magnitude, cmap='gray')
    plt.title("FFT")

    # Edges
    plt.subplot(1,4,3)
    plt.imshow(edges, cmap='gray')
    plt.title("Edges")

    # Overlay
    if overlay is not None:
        plt.subplot(1,4,4)
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.title("Tampered Region")

    plt.suptitle(f"Prediction: {label}", fontsize=14)
    plt.show()


# -------------------------------
# TEST IMAGE (contains path to a test image in data\fake folder)
# -------------------------------
predict_image(r"B:\image-forgery-detection-cv\data\fake\copymove\00048.png")