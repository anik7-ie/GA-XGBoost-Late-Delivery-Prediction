# GA-XGBoost-Late-Delivery-Prediction
Official code repository for A Two-Tier Leakage-Aware Framework for Late Delivery Risk Prediction using GA, XGBoost, and SHAP
# A Two-Tier Leakage-Aware Framework for Late Delivery Risk Prediction

This repository contains the implementation of a hybrid machine learning pipeline for predicting late delivery risk using the DataCo Smart Supply Chain dataset.

## Framework Overview
* **Feature Selection:** Genetic Algorithm (GA) wrapper feature selection.
* **Model:** XGBoost Classifier with SMOTE class balancing and hyperparameter tuning.
* **Explainability:** Global and local SHAP analysis.

## Dataset
The dataset used in this study is the **DataCo Smart Supply Chain Dataset**, available publicly on Kaggle.

## Required Libraries
* `python 3.8+`
* `pandas`
* `numpy`
* `scikit-learn`
* `xgboost`
* `shap`
* `imbalanced-learn`

## How to Run
1. Download the DataCo dataset from Kaggle.
2. Place the dataset in the working directory.
3. Run `jupyter notebook` and execute the main pipeline file.
