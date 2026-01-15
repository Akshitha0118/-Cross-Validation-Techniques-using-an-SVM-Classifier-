# -Cross-Validation-Techniques-using-an-SVM-Classifier-
This  project focused on evaluating and validating a Support Vector Machine (SVM) classifier using multiple cross-validation and hyperparameter tuning techniques

# 🚀 SVM Classification Dashboard (Streamlit)

An **end-to-end Machine Learning Classification project** built using **Support Vector Machines (SVM)** and deployed with **Streamlit**.  
This interactive dashboard demonstrates the complete ML workflow including **data preprocessing, model training, evaluation, cross-validation, and hyperparameter tuning**.

---

## 📌 Project Overview

This project uses the **Social Network Ads dataset** to predict whether a user will purchase a product based on:

- **Age**
- **Estimated Salary**

The model is trained using **Support Vector Classification (SVC)** with an **RBF kernel**, and performance is evaluated using multiple validation strategies.

---

## 🧠 Key Features

✅ Clean & Interactive **Streamlit Dashboard**  
✅ Feature Scaling using **StandardScaler**  
✅ **SVM (RBF Kernel)** Classification  
✅ Performance Metrics:
- Accuracy
- Training Accuracy (Bias)
- Testing Accuracy (Variance)
- Confusion Matrix  
✅ **Cross Validation (K-Fold)**  
✅ **Grid Search CV** for optimal hyperparameters  
✅ **Randomized Search CV** for faster optimization  
✅ Custom UI with **HTML & CSS styling**

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit**
- **NumPy**
- **Pandas**
- **Matplotlib**
- **Scikit-learn**
- **SciPy**

---

## 📂 Dataset

**Social_Network_Ads.csv**

**Features Used:**
- Age
- Estimated Salary

**Target Variable:**
- Purchased (0 / 1)

---

## ⚙️ Machine Learning Workflow

1. Data Loading & Feature Selection  
2. Train-Test Split (80% / 20%)  
3. Feature Scaling  
4. Model Training using SVM (RBF Kernel)  
5. Model Evaluation:
   - Accuracy
   - Bias & Variance
   - Confusion Matrix  
6. Cross Validation (8-Fold)  
7. Hyperparameter Tuning:
   - GridSearchCV
   - RandomizedSearchCV  

---

## 📊 Model Performance (Sample Output)

- **Accuracy:** ~95%
- **Training Accuracy (Bias):** ~90%
- **Testing Accuracy (Variance):** ~95%
- **Cross Validation Accuracy:** ~89%
- **Best Grid Search Parameters:**  
  ```python
  {'C': 1, 'kernel': 'rbf'}

---

## Best Random Search Parameters:

{'C': 57.71, 'gamma': 0.28, 'kernel': 'rbf'}


## 🖥️ Dashboard Preview

- The Streamlit dashboard displays:
- Performance cards (Accuracy, Bias, Variance)
- Confusion Matrix Visualization
- Cross Validation Score
- Best Hyperparameters from Grid & Random Search

## 📌 Future Enhancements

- ROC Curve & AUC Score
- Interactive Hyperparameter Controls
- Model Comparison (Logistic, Random Forest, XGBoost)
- Dataset Upload Option
- Deployment on Streamlit Cloud
