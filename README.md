# Loan Approval Prediction using Machine Learning 🏦📊

## Project Overview
This project focuses on predicting **loan approval status** based on an applicant’s demographic, financial, and credit-related information.  
Using multiple machine learning classification models, the project aims to identify key factors influencing loan decisions and compare model performance.

This project is part of the **PGDM – Big Data Analytics** academic curriculum.

---
## Objective
- Predict whether a loan application will be **Approved or Not Approved**
- Perform **data cleaning, preprocessing, and feature engineering**
- Compare multiple machine learning models
- Identify **important features** affecting loan approval decisions

---
## Dataset Description
The dataset contains **45,000 loan records** with the following key features:
- Applicant age, gender, education
- Income and employment experience
- Loan amount and loan intent
- Interest rate and loan-to-income ratio
- Credit score and credit history length
- Previous loan default history
- Target variable: **loan_status** (0 = Not Approved, 1 = Approved)

The dataset was cleaned to remove duplicates, missing values, and illogical records (e.g., employment experience greater than age). :contentReference[oaicite:0]{index=0}

---
## Data Preprocessing
- Removed duplicate and inconsistent records
- Checked and handled missing values
- One-hot encoded categorical variables
- Feature scaling using **StandardScaler**
- Train-test split (70% training, 30% testing)

---
## Exploratory Data Analysis (EDA)
Key visualizations include:
- Loan status distribution
- Credit score vs loan approval
- Applicant income distribution
- Income vs loan amount scatter plots
- Correlation heatmap of numerical features

EDA helped identify strong predictors such as **credit score, income, and loan amount**.

---
## Models Implemented
The following machine learning models were trained and evaluated:

### Logistic Regression
- Baseline classification model
- Performance evaluated using accuracy, confusion matrix, ROC-AUC
- Train vs test accuracy comparison

### Decision Tree Classifier
- Captures non-linear relationships
- Visualized performance using confusion matrix and ROC curve
- Compared training and testing accuracy

### Random Forest Classifier (Best Model)
- Ensemble learning approach
- Achieved the **highest performance**
- Feature importance analysis performed
- ROC-AUC curve used for evaluation

---
## Model Evaluation Metrics
- Accuracy Score
- Confusion Matrix
- Precision, Recall, F1-score
- ROC-AUC Curve
- Train vs Test Accuracy comparison

---
## Key Findings
- **Random Forest** performed best among all models
- Credit score, income, loan amount, and loan percent income are top predictors
- Proper feature engineering significantly improved model performance
- Ensemble models reduced overfitting compared to Decision Tree

---
## Tools & Technologies Used
- Python
- Google Colab
- Pandas, NumPy
- Matplotlib & Seaborn
- Scikit-learn

---

## 📁 Repository Structure
