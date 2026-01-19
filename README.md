# 💓 AI-Powered Heart Disease Risk Prediction System (UCI)

![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Spaces-yellow)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-purple)
![License](https://img.shields.io/badge/License-MIT-green)
![Live Demo](https://img.shields.io/badge/Live-Demo-success)

---

## 📌 Project Overview

This project presents an **end-to-end Heart Disease Risk Prediction System** built using classical machine learning techniques and deployed as an interactive **Streamlit web application**. The model predicts the **probability of heart disease** based on patient clinical attributes such as age, cholesterol level, resting blood pressure, heart rate, and ECG results.

The system emphasizes:
- Clinical interpretability
- Robust preprocessing & EDA
- Transparent model evaluation
- Explainable AI using SHAP

**Dataset Source (UCI):**  
[Kaggle Dataset UCI](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)

---

## 🧠 System Architecture

1. **Data Ingestion**
   - Loads UCI Heart Disease dataset
   - Handles numerical and categorical features

2. **Exploratory Data Analysis**
   - Histograms and boxplots
   - Correlation heatmap
   - Missing value analysis
   - Target distribution

3. **Preprocessing**
   - Encoding categorical features
   - Feature scaling
   - Pipeline-based transformations (no data leakage)

4. **Model Training**
   - Scikit-learn classifier
   - Optimized using ROC-AUC and F1-score

5. **Evaluation**
   - Confusion Matrix
   - ROC Curve
   - Precision-Recall Curve

6. **Explainability**
   - SHAP for global and local interpretability

7. **Deployment**
   - Streamlit app
   - Dockerized
   - Hosted on Hugging Face Spaces

---

## 🛠️ Technologies Used

- Python 3.9+
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- SHAP
- Streamlit
- Joblib
- Docker

---

## 📊 Dataset Features

| Feature | Description |
|------|------------|
| age | Age of the patient |
| sex | Sex (Male/Female) |
| cp | Chest pain type |
| trestbps | Resting blood pressure |
| chol | Serum cholesterol |
| fbs | Fasting blood sugar |
| restecg | Resting ECG |
| thalch | Max heart rate |
| exang | Exercise induced angina |
| oldpeak | ST depression |
| slope | Slope of ST segment |
| ca | Major vessels count |
| thal | Thalassemia |
| target | Heart disease (0/1) |

---

## 📈 Exploratory Data Analysis

### Target Distribution  

![Target Distribution](output/preprocessing/target_distribution.png)

### Missing Values  

![Missing Values](output/preprocessing/missing_values.png)

### Correlation Heatmap  

![Correlation](output/preprocessing/correlation.png)

### Feature Distributions

**Age**  

![Age Histogram](output/preprocessing/age_hist.png)  

![Age Boxplot](output/preprocessing/age_boxplot.png)

**Cholesterol**  

![Chol Histogram](output/preprocessing/chol_hist.png)  

![Chol Boxplot](output/preprocessing/chol_boxplot.png)

**Max Heart Rate (thalch)**  

![Thalach Histogram](output/preprocessing/thalch_hist.png)  

![Thalach Boxplot](output/preprocessing/thalch_boxplot.png)  


**Resting Blood Pressure (trestbps)**  

![Trestbps Histogram](output/preprocessing/trestbps_hist.png)  

![Trestbps Boxplot](output/preprocessing/trestbps_boxplot.png)  


**Oldpeak**  

![Oldpeak Histogram](output/preprocessing/oldpeak_hist.png)  

![Oldpeak Boxplot](output/preprocessing/oldpeak_boxplot.png)  


**CA (Major Vessels)**  

![CA Histogram](output/preprocessing/ca_hist.png)  

![CA Boxplot](output/preprocessing/ca_boxplot.png)  


---

## 📉 Model Evaluation

### Confusion Matrix  

![Confusion Matrix](output/evaluation/confusion_matrix.png)

### ROC Curve  

![ROC Curve](output/evaluation/roc_curve.png)

### Precision-Recall Curve  

![PR Curve](output/evaluation/pr_curve.png)

---

## 🚀 Live Application

**Hugging Face Spaces:**  
https://huggingface.co/spaces/gouthamkrishna404/Heart-Disease-Risk-Prediction-UCI
Currently down

