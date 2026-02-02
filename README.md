🌍 Predicting Exoplanet Habitability Using Machine Learning
📌 Project Overview

This project focuses on the early-stage development of a machine learning pipeline for predicting exoplanet habitability.
The work covers data understanding, preprocessing, and machine learning dataset preparation, following scientific and industry-standard practices.

The objective of the current phase (Weeks 1–3) is to ensure high-quality, leakage-free data before model training.

🎯 Objectives (Weeks 1–3)
Understand exoplanet and host star characteristics
Clean and preprocess raw astronomical data
Engineer domain-driven habitability features
Prepare machine learning–ready datasets
Avoid data leakage through proper preprocessing pipelines

🗂️ Project Structure
ExoHabit-AI/
│
├── data/
│   ├── raw/
│   │   └── exoplanet_raw.csv
│   │
│   ├── processed/
│   │   ├── exoplanet_cleaned.csv
│   │   ├── X_train.csv
│   │   ├── X_test.csv
│   │   ├── y_train.csv
│   │   ├── y_test.csv
│   │   └── preprocessing_pipeline.pkl
│
├── notebooks/
│   ├── Week1_Data_Understanding.ipynb
│   ├── Week2_Data_Cleaning_Feature_Engineering.ipynb
│   └── Week3_ML_Dataset_Preparation.ipynb
│
├── README.md
└── .gitignore

📅 Weekly Breakdown
🔹 Week 1 — Data Collection & Understanding
Explored exoplanet and host star datasets
Studied feature distributions, correlations, and observational biases
Created astronomy-driven visualizations
Documented initial scientific observations
Output:
Loaded dataset
Exploratory visualizations
Initial observations

🔹 Week 2 — Data Cleaning & Feature Engineering
Handled missing values using robust statistical methods
Analyzed and retained astronomical outliers
Encoded stellar spectral class information
Normalized numerical features
Engineered custom features:
Habitability Score Index
Stellar Compatibility Index
Validated data quality using visualizations
Output:
Cleaned dataset
Feature-engineered dataset
Data validation plots

🔹 Week 3 — Machine Learning Dataset Preparation
Selected important features using correlation analysis and domain knowledge
Defined a binary target variable (Habitable / Not Habitable)
Performed an 80:20 stratified train–test split
Built a unified preprocessing pipeline:
Scaling
Encoding
Ensured no data leakage by fitting preprocessing only on training data
Saved ML-ready datasets and preprocessing pipeline
Output:
Final feature matrix (X) and target vector (y)
Train–test datasets
ML-ready preprocessing pipeline

🛠️ Tools & Technologies Used
Python
Pandas, NumPy
Matplotlib, Seaborn
Scikit-learn
Joblib
Git & GitHub Desktop

🔮 Next Phase (Planned)
Machine learning model training and evaluation
Exoplanet habitability prediction and ranking
(Covered in subsequent weeks)
