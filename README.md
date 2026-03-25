# Image Forgery Detection using Classical Computer Vision

## 1. Introduction

This project detects whether an image has been tampered with using classical computer vision techniques.  
The approach is based on extracting features from images such as frequency information, edges, texture, and color, and then using a machine learning model to classify images as **real** or **tampered**.

The project is designed to run completely from the command line.



## 2. Project Structure
```
image-forgery-detection-cv/
│
├── data/
│   ├── real/        # Authentic images
│   ├── fake/        # Tampered images (with subfolders)
│   ├── mask/        # Ground truth masks (for visualization)
│
├── src/
│   ├── features.py  # Feature extraction
│   ├── train.py     # Model training
│   ├── predict.py   # Prediction + visualization
│   ├── utils.py     # Helper functions
│
├── outputs/         # Optional saved outputs
├── requirements.txt
└── README.md
```



## 3. Requirements

- Python 3.8 or higher
- pip (Python package manager)





## 4. Environment Setup

### Step 1: Clone the repository
```bash
git clone https://github.com/sagar-24bytes/image-forgery-detection-cv
cd image-forgery-detection-cv
```

### Step 2: Create a virtual environment (recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install dependencies
```bash
pip install -r requirements.txt
```



## 5. Dataset Setup

The dataset used in this project is sourced from Kaggle:

Dataset link: https://www.kaggle.com/datasets/minghaoli99/image-tampering-dataset?select=Dataset

After downloading, extract the dataset and place it inside the `data/` folder with the following structure:

```
data/
├── real/
├── fake/
│   ├── copymove/
│   ├── inpainting/
│   └── splice/
└── mask/
```

- `real/` — contains original, authentic images  
- `fake/` — contains tampered images organized by type  
- `mask/` — contains ground truth masks (optional, used only for visualization)



## 6. How to Run the Project

### Step 1: Train the model
```bash
python -m src.train
```

This will:
- Load the dataset
- Extract features
- Train the SVM model
- Save `model.pkl` and `scaler.pkl`

### Step 2: Run prediction
```bash
python -m src.predict
```

This will:
- Load the trained model
- Run prediction on a test image
- Display the original image, FFT spectrum, edge map, and tampered region overlay (if mask exists)



## 7. How the System Works

1. Images are resized and preprocessed
2. Features are extracted:
   - **Frequency features** (FFT)
   - **Edge features** (Canny)
   - **Texture features** (HOG)
   - **Color histogram**
3. Features are normalized using a scaler
4. An SVM classifier predicts whether the image is **real** or **tampered**



## 8. Output

For each test image, the system displays:

- Original image
- Frequency spectrum (FFT)
- Edge detection output
- Tampered region overlay (if mask is available)

During training:
- Accuracy score is printed to the terminal
- Confusion matrix is displayed

## Sample Output

Below is an example of the prediction result:

- Original image  
- FFT spectrum  
- Edge detection  
- Highlighted tampered region  

<img width="900" height="350" alt="sample_result" src="https://github.com/user-attachments/assets/10a5a2de-f766-4e9d-acb3-1e68bacc8af3" />

The model predicts the image as **tampered** based on inconsistencies observed in different feature representations.

- The **FFT spectrum** highlights irregular frequency patterns that may indicate manipulation.  
- The **edge map** shows discontinuities and unnatural boundaries introduced during editing.  
- The **tampered region overlay** (highlighted area) corresponds to the ground truth mask, indicating where the image was modified.

Together, these visual cues help explain why the model classifies the image as tampered.

## Model Evaluation

A confusion matrix is used to analyze the model’s performance.

<img width="450" height="500" alt="confusion_matrix" src="https://github.com/user-attachments/assets/be180759-90fa-4144-9ea9-eedbe5ee2c8c" />

The results show that the model is able to identify both real and tampered images, but with moderate accuracy. This is expected due to the use of classical feature-based methods and the complexity of image forgery detection.

## 9. Notes and Limitations

- Accuracy is moderate due to use of classical (non-deep learning) methods
- Performance depends on dataset quality and size
- This project focuses on explainability over high accuracy



## 10. Running from Command Line

This project is fully executable via terminal — no GUI setup required:
```bash
python -m src.train
python -m src.predict
```



## 11. Conclusion

This project demonstrates how image processing techniques such as frequency analysis, edge detection, and texture features can be used to detect image tampering. While the approach is classical, it provides a clear and interpretable pipeline for understanding how visual cues can indicate manipulation.
