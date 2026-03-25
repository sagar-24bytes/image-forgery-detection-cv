import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

from src.features import extract_features
from src.utils import load_images

# -------------------------------
# Load data (INCREASED SIZE)
# -------------------------------
real_imgs, real_labels = load_images("data/real", 0, limit=100)
fake_imgs, fake_labels = load_images("data/fake", 1, limit=100)

images = real_imgs + fake_imgs
labels = real_labels + fake_labels

# -------------------------------
# Feature extraction
# -------------------------------
X = []
for img in images:
    feat = extract_features(img)
    X.append(feat)

X = np.array(X)
y = np.array(labels)

print("Dataset shape:", X.shape)

# -------------------------------
# NORMALIZATION (IMPORTANT)
# -------------------------------
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Save scaler
joblib.dump(scaler, "scaler.pkl")

from sklearn.utils import shuffle
# # shuffling data 
# X, y = shuffle(X, y, random_state=42)
# -------------------------------
# Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# MODEL (RBF Kernel)
# -------------------------------
model = SVC(kernel='rbf', C=5)

model.fit(X_train, y_train)

# -------------------------------
# Evaluation
# -------------------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("Accuracy:", acc)

# -------------------------------
# Save model
# -------------------------------
joblib.dump(model, "model.pkl")

print("Model saved as model.pkl")
print("Scaler saved as scaler.pkl")

# confusion matrix

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()