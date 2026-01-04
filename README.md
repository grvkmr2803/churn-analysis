# Customer Churn Prediction with Explainable AI

This project is an end-to-end machine learning application that predicts customer churn for a telecom company and explains individual predictions using Explainable AI (SHAP).  
The system is designed to be interpretable, interactive, and deployable as a real-world decision-support tool.

---

## Problem Statement

Customer churn is a critical business problem in subscription-based industries.  
The goal of this project is to:
- Predict the probability that a customer will churn
- Explain *why* the model made a particular prediction
- Translate model insights into actionable business recommendations

---

## Solution Overview

The application uses an XGBoost classification model combined with SHAP (SHapley Additive exPlanations) to provide transparent, customer-level explanations.

Key features:
- Probability-based churn prediction
- Local explainability using SHAP waterfall plots
- Interactive Streamlit dashboard for real-time analysis
- Business-oriented insights derived from model explanations

---

## Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- SHAP
- Streamlit
- Matplotlib
- Joblib

---

## Machine Learning Pipeline

1. Data Cleaning and Preprocessing
   - Handled missing values in numerical features
   - Removed non-informative identifiers
   - Applied OneHotEncoding to categorical variables to avoid artificial ordinal relationships

2. Model Training
   - Trained an XGBoost classifier
   - Used a Scikit-learn Pipeline for preprocessing and modeling
   - Evaluated performance using ROC-AUC

3. Explainability
   - Used SHAP TreeExplainer with background data
   - Generated local explanations for individual customer predictions
   - Preserved feature names after preprocessing for interpretability

---

## Application Features

- Interactive sidebar for customer profile input
- KPI card displaying churn probability
- Risk level classification (Low / Medium / High)
- SHAP-based explanation of prediction drivers
- Natural-language interpretation of top contributing features
- Business recommendations based on model output

---

## Project Structure

.
├── app.py # Streamlit application
├── train_model.py # Model training pipeline
├── churn_model.pkl # Trained ML pipeline
├── X_train_raw.csv # Training data for SHAP background
├── requirements.txt # Project dependencies
└── README.md



---

