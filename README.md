# ❤️ Heart Disease Prediction using Machine Learning

## 📌 Project Overview

This project aims to predict the likelihood of heart disease using Machine Learning techniques based on patient medical attributes. The model analyzes clinical features such as age, chest pain type, cholesterol level, blood pressure, heart rate, and other medical indicators to classify whether a patient is at risk of heart disease.

The project uses the **UCI Heart Disease (Cleveland)** dataset and compares multiple classification algorithms to identify the best-performing model. After model evaluation and hyperparameter tuning, the **Random Forest Classifier** was selected as the final model due to its superior predictive performance.

---

# 🎯 Objectives

* Predict whether a patient has heart disease.
* Compare multiple Machine Learning algorithms.
* Improve model performance using Hyperparameter Tuning.
* Save the trained model for future predictions.

---

# 📂 Dataset

* **Dataset:** UCI Heart Disease (Cleveland)
* **Total Records:** 303
* **Features:** 13
* **Target Variable:**

  * **0** → No Heart Disease
  * **1** → Heart Disease

---

# 📊 Features Used

* Age
* Sex
* Chest Pain Type (cp)
* Resting Blood Pressure (trestbps)
* Cholesterol (chol)
* Fasting Blood Sugar (fbs)
* Resting ECG (restecg)
* Maximum Heart Rate (thalach)
* Exercise Induced Angina (exang)
* ST Depression (oldpeak)
* Slope of ST Segment (slope)
* Number of Major Vessels (ca)
* Thalassemia (thal)

---

# 🔍 Data Preprocessing

The following preprocessing steps were performed:

* Missing value handling using **Median Imputation**
* Conversion of categorical values into numerical format
* Binary conversion of the target variable
* Feature Scaling using **StandardScaler**
* Train-Test Split (80:20)

---

# 🤖 Machine Learning Models

The following models were trained and evaluated:

* Logistic Regression
* Support Vector Machine (SVM)
* Random Forest Classifier
* XGBoost Classifier

---

# 📈 Model Performance

| Model                   |     Accuracy |
| ----------------------- | -----------: |
| Logistic Regression     |   **86.89%** |
| Support Vector Machine  |   **85.25%** |
| Random Forest           |   **88.52%** |
| **Tuned Random Forest** | **90.16%** ✅ |

---

# 🏆 Final Model

The Random Forest model was further optimized using **GridSearchCV**.

### Best Parameters

```python
{
    'max_depth': 5,
    'min_samples_leaf': 1,
    'min_samples_split': 5,
    'n_estimators': 100
}
```

### Final Performance

* **Algorithm:** Tuned Random Forest
* **Accuracy:** **90.16%**
* **ROC-AUC Score:** **94.81%**

The tuned Random Forest model was selected as the final model because it achieved the highest prediction accuracy while maintaining excellent classification performance.

---

# 📦 Saved Model

The trained model is saved as:

```
heart_disease_model.pkl
```

Feature scaling object:

```
scaler.pkl
```

These files can be loaded later to perform predictions on new patient data without retraining the model.

---

# 📚 Python Libraries Used

* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* Joblib

---

# 🚀 Workflow

```
Load Dataset
      ↓
Data Cleaning
      ↓
Missing Value Handling
      ↓
Exploratory Data Analysis (EDA)
      ↓
Feature Scaling
      ↓
Train-Test Split
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Hyperparameter Tuning
      ↓
Best Model Selection
      ↓
Model Serialization (.pkl)
```

---

# 👨‍💻 Developed By

**Akshat Goyal**

Mathematics & Computing Engineering Student

---

# 🙏 Acknowledgements

* CodeAlpha Machine Learning Internship
* UCI Machine Learning Repository
* Scikit-learn Documentation
* XGBoost Documentation
