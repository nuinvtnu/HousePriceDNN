# Lightweight Deep Neural Networks for House Price Prediction: Performance Evaluation and SHAP-Based Interpretability Analysis

## Overview

This project implements and compares several machine learning and deep learning models for house price prediction using the King County housing dataset.

Implemented models:
- Linear Regression (LR)
- Bayesian Ridge Regression (BR)
- K-Nearest Neighbors (KNN)
- Random Forest (RF)
- Support Vector Regression (SVR)
- XGBoost
- Deep Neural Network (DNN)

## Project Structure

Code_predict_house_price/
├── dataset/
│   └── kc_house_data.csv
├── save_result.py
├── main_regression.py
└── result/
    ├── LinearRegression/
    ├── BayesianRidge/
    ├── RandomForest/
    ├── KNN/
    ├── SVR/
    ├── XGBoost/
    ├── DNN/
    └── Summary_Regression.xlsx

## Installation

pip install numpy pandas matplotlib seaborn scikit-learn tensorflow xgboost openpyxl

## Running

python main_regression.py

## Evaluation Metrics

- MSE
- RMSE
- MAE
- R²

