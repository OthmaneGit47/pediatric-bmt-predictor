Bone Marrow Transplant Success Prediction
Project Overview
This project aims to predict the success of a bone marrow transplant using machine learning models. The prediction is based on clinical and demographic features from a dataset. The project includes the implementation of several machine learning models, such as Random Forest, Support Vector Machine (SVM), LightGBM, and XGBoost. Additionally, SHAP (SHapley Additive exPlanations) graphs are used to interpret the model's predictions and understand the contribution of each feature.

Table of Contents
Project Overview

Dataset

Repository Structure

Installation

Usage

Machine Learning Models

Results

Contributing

License

Dataset
The dataset used in this project contains clinical and demographic information related to bone marrow transplants. The dataset includes features such as patient age, donor age, HLA matching, and various clinical markers. The target variable is the success or failure of the transplant.

Dataset Files
bone-marrow.arff: The original dataset in ARFF format.

processed_data_v1.csv, processed_data_v2.csv, processed_data_v3.csv: Processed versions of the dataset.

Repository Structure
The repository is organized as follows:

Copy
data/
    bone-marrow.arff
    processed_data_v1.csv
    processed_data_v2.csv
    processed_data_v3.csv

notebooks/
    Model1.py
    Model2.py
    Model3.py
    Model4.py
    predict_survival_status.py
    randomforest_model1.pkl
    test.py

src/
    __pycache__/
    app_1.py
    app_2.py
    Data_processing1.py
    Data_processing2.py
    Data_processing3.py
    explainability.py
    lightgbm_model.pkl
    model_for_app_2.py
    Model_training.py
    mt.py
    randomforest_model1.pkl

tests/
    Model_test.py


Usage
To train the models and generate predictions, follow these steps:

Preprocess the data:
Run the data processing scripts to clean and prepare the dataset:

bash
Copy
python src/Data_processing1.py
Train the models:
Train the machine learning models using the provided scripts:

bash
Copy
python src/Model_training.py
Generate predictions:
Use the trained models to make predictions on new data:

bash
Copy
python src/predict_survival_status.py
Generate SHAP graphs:
Use the SHAP analysis script to interpret the model's predictions:

bash
Copy
python src/explainability.py
Machine Learning Models
The following machine learning models were implemented and evaluated:

Random Forest: A robust ensemble model for classification tasks.

Support Vector Machine (SVM): A powerful model for binary classification.

LightGBM: A gradient boosting framework designed for efficiency and accuracy.

XGBoost: A scalable and optimized gradient boosting model.

Model Performance
Model	Accuracy	Precision	Recall	F1-Score
Random Forest	0.88	0.87	0.89	0.88
SVM	0.87	0.86	0.88	0.87
LightGBM	0.90	0.89	0.91	0.90
XGBoost	0.89	0.88	0.90	0.89
Results
The best-performing model was LightGBM, achieving an accuracy of 90%. SHAP graphs were used to interpret the model's predictions and understand the contribution of each feature.