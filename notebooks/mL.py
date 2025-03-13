import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Load your dataset (adjust this according to your dataset file)
# For example, assume the dataset is in a CSV file called 'pediatric_bone_marrow.csv'
df = pd.read_csv('cleaned_dataset.csv')

# Step 2: Preprocess the data (handle missing values, encode categorical features, etc.)
# Let's assume that your target column is 'survival_status' and the rest are features
# Adjust the following lines based on the dataset structure




# Step 3: Separate features (X) and target variable (y)
X = df.drop('survival_status', axis=1)  # All columns except 'survival_status'
y = df['survival_status']  # The target column

# Step 4: Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Initialize the RandomForestClassifier model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Step 6: Train the model
rf_model.fit(X_train, y_train)

# Step 7: Make predictions on the test data
y_pred = rf_model.predict(X_test)

# Step 8: Evaluate the model's performance
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# If needed, you can also access feature importance
feature_importance = pd.DataFrame(rf_model.feature_importances_, index=X.columns, columns=['Importance'])
print("\nFeature Importance:\n", feature_importance.to_string())
