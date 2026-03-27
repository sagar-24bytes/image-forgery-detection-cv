import cv2
import numpy as np
from skimage.feature import hog

def preprocess(image):
    image = cv2.resize(image, (128, 128))
    return image

# FFT (simple stats instead of full flatten)
def extract_fft(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    magnitude = np.log(np.abs(f) + 1)

    return np.array([
        np.mean(magnitude),
        np.std(magnitude)
    ])

# Edge density (not full edges)
def extract_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)

    return np.array([
        np.mean(edges),
        np.std(edges)
    ])

# HOG (reduced)
def extract_hog(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features = hog(gray, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), visualize=False)

    return features[:200]

# Color histogram 
def extract_color(image):
    hist = cv2.calcHist([image], [0,1,2], None,
                        [8,8,8], [0,256,0,256,0,256])
    hist = cv2.normalize(hist, hist).flatten()

    return hist[:200]

def extract_features(image):
    image = preprocess(image)

    fft = extract_fft(image)
    edges = extract_edges(image)
    hog_feat = extract_hog(image)
    color_feat = extract_color(image)

    return np.hstack([fft, edges, hog_feat, color_feat])