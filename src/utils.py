import os
import cv2

def load_images(folder, label, limit=100):
    data = []
    labels = []

    count = 0

    for root, dirs, files in os.walk(folder):
        for file in files:
            if count >= limit:
                break

            path = os.path.join(root, file)
            img = cv2.imread(path)

            if img is None:
                continue

            data.append(img)
            labels.append(label)
            count += 1

    return data, labels