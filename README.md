Bone Marrow Transplant Success Prediction Project Overview This project aims to predict the success of a bone marrow transplant using machine learning models. The prediction is based on clinical and demographic features from a dataset. The project includes the implementation of several machine learning models, such as Random Forest, Support Vector Machine (SVM), LightGBM, and XGBoost. Additionally, SHAP (SHapley Additive exPlanations) graphs are used to interpret the model's predictions and understand the contribution of each feature.

Table of Contents Project Overview

Dataset:
The dataset was initially irrelevent, it had missing values, imbalances issues, outliers and unnecessary features. Therefore we needed to make it more suitable by making a Data_processing file, this python file makes a sequence of processing on the data :
1-Missing values: We noticed that there was only a small number of missing values, with a maximum of 16% missing values in a single feature, therefore there was no need for deleting some rows thus reducing the size of the dataset, which is not advised for the model training efficiency, so the solution that seemed reasonable was to replace the missing values with the most frequent(in the 'object' type cases) or the mean value(in the 'number' type cases)
2-'object' type values: Since it's not relevant for model training we had to encode them into 'numerical' types so it can have meaning for the models
3-Outliers: Some features had very differentiated values, and this is not appropriate for the model training because it can't really learn from this feature, so we handled this problem by checking for outliers first then replacing them with the median value using the bounds method
4-Imbalanced data: The dataset was initially imbalanced, meaning there was a clear difference between the classes(the targets), therefore we had to use the over sampling method to create a balanced data set that won't cause the model to be biased.
5-Correlation: There were some features that have a high correlation factor, meaning they depend on each other somehow, these kind of features don't attribute to the model's accuracy but might be somewhat misguiding, so we had to remove all the features that have a correlation factor higher than 0.9 with another feature
6-Optimization: We created a function named optimize_data to optimize the size of the dataset before saving it.




Installation:see requirements.txt

Usage:
This:
git clone https://github.com/Ali-m-1/predict-bmt-predictor
cd your-repo
Or:
docker build -t flask-app .
Then this:
docker save -o flask-app.tar flask-app
docker load -i flask-app.tar
docker run -d -p 5000:5000 flask-app

Make sure to have docker installed

Machine Learning Models:
We tested 4 models on the processed_data file, the 1st one was the best perfoming because it had the best precision/recall values among the rest and the highest f1-score : 0.97, compared to the other three.
Find the models in notebooks

Results:
The testing of the Random Forest Classifier gave us some explanations of the most influencing factors using SHAP eplainability, it stated that the most influencing feature was the 'survival_time' meaning the more time the patient survived the more it can signify he will live.
Find the shap graphs in src/templates or see the code in src/explainability

Contributing:
Rabbah Mohamed Ali : mohamedali.rabbah@centrale-casa.ma
Barhoud Othmane : othmane.barhoud@centrale-casa.ma
Serbouti Youssef : youssef.serbouti@centrale-casa.ma
Boulezhar Chadi : chadi.boulezhar@centrale-casa.ma


