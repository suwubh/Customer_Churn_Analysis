Customer Churn Analysis
📌 Overview

This project focuses on analyzing and predicting customer churn using the Telco Customer Churn dataset. The workflow covers exploratory data analysis (EDA), model building, and interactive visualization through Power BI. The aim is to identify key factors influencing churn and build machine learning models to predict whether a customer is likely to leave.

📂 Repository Structure

Churn Analysis - EDA.ipynb → Data cleaning, visualization, and exploratory insights.

Churn Analysis - Model Building.ipynb → Model training and evaluation (Logistic Regression, Random Forest, etc.).

Customer Churn Knowledge.py → Supporting scripts for analysis/modeling.

WA_Fn-UseC_-Telco-Customer-Churn.csv / first_telc.csv / tel_churn.csv → Datasets used in the project.

churm analysis.pbix → Power BI dashboard for interactive reporting.

model.sav → Saved machine learning model for predictions.

⚙️ Getting Started
Prerequisites

Python 3.x

Required libraries:

pip install pandas numpy matplotlib seaborn scikit-learn


Power BI Desktop (for .pbix file)

Setup

Clone the repository:

git clone https://github.com/suwubh/Customer_Churn_Analysis.git
cd Customer_Churn_Analysis


Run the EDA notebook to explore the data.

Use the Model Building notebook to train and evaluate models.

Load model.sav for predictions:

import joblib
model = joblib.load("model.sav")


Open churm analysis.pbix in Power BI Desktop to explore dashboards.

📊 Results & Insights

Identified important churn factors such as contract type, tenure, monthly charges, and payment method.

Built multiple ML models for churn prediction (best model achieved strong accuracy).

Created an interactive Power BI dashboard to visualize customer churn trends.

🚀 Future Improvements

Hyperparameter tuning for better accuracy.

Deploy model via Flask/Django for real-time prediction.

Automate Power BI dashboard refresh.

Add a requirements.txt for easier setup.
