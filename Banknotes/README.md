# Banknote Authentication – k-NN vs Logistic Regression

This project uses the **UCI Banknote Authentication Dataset** to detect counterfeit banknotes
using simple rules and machine learning models.

## Dataset

Source: [UCI Banknote Authentication Dataset](https://archive.ics.uci.edu/ml/datasets/banknote+authentication)

Each banknote is described by four continuous features extracted from 400×400 gray-scale images
via a wavelet transform:

1. `f1` – variance of the transformed image  
2. `f2` – skewness  
3. `f3` – curtosis  
4. `f4` – entropy  
5. `class` – 0 = genuine, 1 = fake  

The goal is to predict `class` using only the four features.

## Methodology

- Exploratory data analysis with pairplots for real (class 0) and fake (class 1) notes  
- **Simple rule-based classifier** using thresholds on variance and curtosis  
- **Train–test split** (50/50), with feature standardization  
- **k-NN classifier**
  - Inner train/validation split to choose optimal k  
  - Evaluation with confusion matrix, TPR, FPR, accuracy  
  - Feature-importance study via leave-one-feature-out accuracy  
- **Logistic regression**
  - Coefficient analysis to interpret feature influence  
  - Evaluation with confusion matrix, TPR, FPR, accuracy  
  - Feature-importance analysis via feature exclusion

## Key Results

- Simple rule-based classifier: ~65% test accuracy, low TPR for fake notes  
- k-NN (k = 3): ~99.7% test accuracy, TPR ≈ 1.0, FPR ≈ 0.005  
- Logistic regression: ~98.4% test accuracy, TPR ≈ 0.993, FPR ≈ 0.024  
- Both models agree that **variance, skewness, and curtosis** are the most important features;  
  entropy adds little additional information.

## Tech Stack

- Python, NumPy, Pandas  
- scikit-learn (k-NN, LogisticRegression, metrics)  
- Matplotlib, Seaborn  
- Jupyter Notebook
