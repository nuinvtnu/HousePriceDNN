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

## Project Structure

```text
HousePriceDNN/
├── dataset/
│   └── kc_house_data.csv
├── result/
├── DNN_5CV.py                 # DNN with 5-fold cross-validation
├── main_DNN.py                # DNN training and independent testing
├── DNN_SHAP.py                # SHAP-based interpretability analysis
├── CNN.py                     # CNN model
├── CNN_5CV.py                 # CNN with 5-fold cross-validation
├── ablation_DNN.py            # DNN ablation study
├── 5fold-ablation-DNN.py      # DNN ablation with 5-fold CV
├── Ablation_ML.py             # ML ablation study
├── 5fold_Ablation-ML.py       # ML ablation with 5-fold CV
└── README.md
```
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
