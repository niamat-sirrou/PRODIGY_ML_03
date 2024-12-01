# Cat and Dog Classification using SVM

This project demonstrates the implementation of a **Support Vector Machine (SVM)** for classifying images of cats and dogs. The project uses Python, along with popular machine learning and data processing libraries such as NumPy, Pandas, Scikit-learn, Matplotlib, Seaborn, and OpenCV.

## Objective

The main objective of this project is to build a binary classification model that distinguishes between images of cats and dogs. The model is trained and evaluated using a dataset of labeled images. The project emphasizes data preprocessing, feature extraction, and the effective application of SVM.

---

## Key Features

- **Image Preprocessing**: 
  - Image resizing and grayscale conversion using OpenCV.
  - Data normalization to improve model performance.
  
- **Feature Extraction**: 
  - Transforming image data into meaningful features suitable for machine learning algorithms.

- **Model Development**:
  - Implementing Support Vector Machines (SVM) using Scikit-learn for binary classification.
  - Hyperparameter tuning for optimizing SVM performance.

- **Visualization**:
  - Data exploration and visualization using Seaborn and Matplotlib.
  - ROC curve and confusion matrix for model evaluation.

---

## Dataset

The dataset used for this project is the [Kaggle Cat and Dog Dataset](https://www.kaggle.com/datasets). The dataset contains thousands of labeled images of cats and dogs.

**Steps to download the dataset:**
1. Visit the Kaggle dataset page and download the dataset.
2. Extract the dataset into a directory (`/data`) in the project folder.
3. Ensure the dataset contains separate folders for `cats` and `dogs` images.

---

## Dependencies

The following Python libraries are required for this project:
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- OpenCV

To install the required packages, run:
```bash
pip install -r requirements.txt
