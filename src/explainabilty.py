import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


df = pd.read_csv('data\Processed_data.csv')
# Sample dataset loading (Replace with actual dataset)
X = df.drop('survival_status', axis=1)  # All columns except 'survival_status'
y = df['survival_status'].values.ravel()

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# SHAP Explanation
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# ✅ DEBUG SHAP VALUES SHAPE
print("X_test shape:", X_test.shape)
print("SHAP values shape:", np.array(shap_values).shape)

# 🔹 If `shap_values` is a list (multi-class case), take the absolute mean across classes
if isinstance(shap_values, list):
    shap_values = np.mean(np.abs(shap_values), axis=0)

# 🔹 Ensure SHAP values are correctly reshaped
shap_values = np.array(shap_values)
if shap_values.ndim == 3:  # If it has an extra dimension, sum over it
    shap_values = shap_values.mean(axis=2)

# ✅ Ensure feature count matches
assert shap_values.shape[1] == X_test.shape[1], "Feature count mismatch!"

# Create DataFrame for SHAP Importance
shap_importance = pd.DataFrame({
    "Feature": X.columns,
    "SHAP Importance": np.abs(shap_values).mean(axis=0)
}).sort_values(by="SHAP Importance", ascending=False)

print("\nFinal SHAP Importance:\n", shap_importance)

# 🎨 SHAP Summary Plot
shap.summary_plot(shap_values, X_test)

# 📊 SHAP Bar Graph
plt.figure(figsize=(10, 6))
plt.barh(shap_importance["Feature"], shap_importance["SHAP Importance"])
plt.xlabel("Mean |SHAP Value|")
plt.ylabel("Feature")
plt.title("Feature Importance via SHAP")
plt.gca().invert_yaxis()
plt.show()