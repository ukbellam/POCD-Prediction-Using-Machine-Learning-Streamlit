# POCD Prediction Using Machine Learning & Streamlit

This project builds a complete machine learning pipeline to predict **Post-Operative Cognitive Dysfunction (POCD)** using structured EHR data.  
Multiple models were trained and evaluated, including Logistic Regression, Random Forest, MLP, Naive Bayes, SVM, and XGBoost (regularization path study).

A **Streamlit web app** is included to allow:
- **CSV-based predictions**
- **Manual single-patient input**
- Downloadable risk outputs

---

## Project Overview

The goal of this project was to train supervised learning models on clinical variables and predict the risk of POCD after surgery.  
The workflow includes:

### **1. Data Preparation**
- Handling missing values  
- Converting timestamps to usable numeric formats  
- Dropping extremely high-cardinality identifiers  
- One-hot encoding categorical features  
- Standardizing numerical variables  
- Splitting into train / validation / test sets  

### **2. Models Trained**
| Model | Notes |
|-------|-------|
| Logistic Regression | Baseline + fully tuned version |
| L1 & L2 Regularization Paths | To study coefficient behavior |
| Random Forest | Tuned via GridSearchCV |
| MLP Classifier | Best-performing model |
| Gaussian Naive Bayes | Generative baseline |


The **MLPClassifier** delivered the strongest PR-AUC and was selected as the final model.

---

## Evaluation Metrics

Because the dataset is imbalanced, the primary metric used is:

- **PR-AUC (Average Precision Score)**
- Precision-Recall curves  
- Classification reports  

This provides a more meaningful evaluation than ROC-AUC in low-incidence medical predictions.

---

## Final Model

The trained final model is: mlp_pocd_pipeline.pkl
This includes:
- StandardScaler  
- Encoded feature structure  
- The trained MLP neural network  

Feature names used during training are stored in: feature_columns.pkl


These two files are required for inference.

---

## 🌐 Streamlit Web App

The repository includes a `app.py` file that provides a simple UI for making predictions.

### Features:
✔ Upload a CSV file with patient records  
✔ OR manually enter patient details  
✔ Automatic preprocessing and feature alignment  
✔ POCD risk score output  
✔ Download predictions as CSV  
