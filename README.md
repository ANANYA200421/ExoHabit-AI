🌌 ExoHabit-AI
Machine Learning Based Exoplanet Habitability Prediction System
🚀 Project Overview

ExoHabit-AI is an end-to-end Machine Learning web application designed to predict and analyze the habitability of exoplanets based on planetary and stellar parameters.

The system integrates:
Data preprocessing pipeline
Machine learning models (Random Forest)
Flask REST API backend
Interactive Plotly dashboard
Responsive frontend interface
Exportable scientific reports

🎯 Objectives
Predict exoplanet habitability probability
Rank exoplanets based on ML confidence
Visualize feature importance and correlations
Provide an interactive scientific dashboard
Enable export of top candidate reports

🧠 Machine Learning Models Used
Logistic Regression (Baseline)
Random Forest (Best Performing)
XGBoost

Evaluation Metrics:
Accuracy
Precision
Recall
F1-Score
ROC-AUC

📊 Interactive Dashboard Features
Feature Importance (Plotly Interactive)
Habitability Probability Distribution
Star–Planet Correlation Heatmap
Downloadable Top 20 Exoplanets Report (Excel)

🖥️ System Architecture
User Input → Flask API → Preprocessing Pipeline → ML Model → Prediction
                                             ↓
                                       Plotly Dashboard

🛠️ Tech Stack
Python
Flask
Scikit-learn
Pandas
Plotly
Bootstrap
JavaScript
HTML/CSS