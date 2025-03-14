#ml3 shap

import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import shap

# Step 1: Load your dataset
df = pd.read_csv('resampled_bone_marrow_dataset.csv')

# Step 2: Preprocess the data (handle missing values, encode categorical features, etc.)
# Adjust the following lines based on the dataset structure

# Step 3: Separate features (X) and target variable (y)
X = df.drop('survival_status', axis=1)  # All columns except 'survival_status'
y = df['survival_status']  # The target column

# Split dataset into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create LightGBM dataset
model = lgb.LGBMClassifier(
    objective='binary',  # Use 'multiclass' if needed
    boosting_type='gbdt',
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    n_estimators=1000
)

# Train the model with early stopping and logging
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],  # Validation set
    eval_metric='logloss',  # Performance metric
    callbacks=[  
        lgb.early_stopping(10),  # Stop if no improvement for 10 rounds
        lgb.log_evaluation(10)   # Print logs every 10 rounds
    ]
)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# SHAP Explanations
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Visualize SHAP values
shap.summary_plot(shap_values, X_test)
shap.force_plot(explainer.expected_value, shap_values[0, :], X_test.iloc[0, :], matplotlib=True)
shap.dependence_plot(0, shap_values, X_test, feature_names=X_test.columns)
shap.plots.waterfall(shap.Explanation(values=shap_values[0, :], base_values=explainer.expected_value, data=X_test.iloc[0, :]))