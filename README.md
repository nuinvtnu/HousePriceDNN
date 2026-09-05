# A Deep Neural Network for House Price Prediction:Comparative Performance Evaluation and SHAP-Based Interpretability Analysis

## Overview
This repository contains the implementation of machine learning and deep learning models for **house price prediction** using the King County housing dataset.
The experiments include:
- Machine learning baseline comparison
- DNN and CNN models
- 5-fold cross-validation
- Ablation studies
- SHAP-based model interpretability

## Models
**Machine Learning:** Linear Regression (LR), Bayesian Ridge (BR), KNN, Random Forest (RF), SVR, and XGBoost.
**Deep Learning:** Deep Neural Network (DNN) and Convolutional Neural Network (CNN).

## Project Structure

HousePriceDNN/
├── dataset/
│   └── kc_house_data.csv
├── result/
├── DNN_5CV.py # 5 fold cross-validation
├── main_DNN.py #buil model and independent test
├── DNN_SHAP.py
├── CNN.py
├── CNN_5CV.py
├── ablation_DNN.py
├── 5fold-ablation-DNN.py
├── Ablation_ML.py
├── 5fold_Ablation-ML.py
└── README.md
## Installation
pip install numpy pandas scikit-learn tensorflow xgboost shap matplotlib
## Running
Run the corresponding script, for example:
python main_DNN.py
python DNN_SHAP.py 
python ablation_DNN.py
python 5fold-ablation-DNN.py

Model DNN predict house: DNN_model.h5 in folder result\DNN_80_20\
```
## Evaluation Metrics
Model performance is evaluated using:MSE, RMSE,MAE,R²
## Interpretability
SHAP (SHapley Additive exPlanations) is used to analyze feature importance and explain the contribution of housing attributes to DNN predictions.
## Results
Experimental results are saved in the `result/` directory.
