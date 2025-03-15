import lightgbm as lgb
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv('data\processed_data_v3.csv')

# Handle missing values (if any)
df.fillna(df.mean(), inplace=True)  # Replace NaN with column mean

# Separate features and target
X = df.drop(columns=['survival_status'])
y = df['survival_status']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train LightGBM model
model = lgb.LGBMClassifier(objective='binary', learning_rate=0.05, num_leaves=31, n_estimators=1000)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='logloss',
          callbacks=[lgb.early_stopping(10), lgb.log_evaluation(10)])

# Model evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# SHAP explanation
explainer = shap.TreeExplainer(model)  # TreeExplainer for LightGBM
shap_values = explainer.shap_values(X_test)

# Fix SHAP values format for binary classification
if isinstance(shap_values, list):
    shap_values = shap_values[1]  # Select positive class if binary classification

# Ensure all features are displayed
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X_test, max_display=X_test.shape[1])

# SHAP Dependence Plot (Example for the most important feature)
shap.dependence_plot(np.argmax(np.abs(shap_values).mean(0)), shap_values, X_test)

# SHAP Waterfall Plot for a single prediction
shap.plots.waterfall(shap.Explanation(values=shap_values[0], base_values=explainer.expected_value[1], data=X_test.iloc[0, :]))
