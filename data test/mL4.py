import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


df = pd.read_csv('cleaned_bone_marrow_dataset.csv')



# Step 3: Separate features (X) and target variable (y)
X = df.drop('survival_status', axis=1)  # All columns except 'survival_status'
y = df['survival_status']  # The target column

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define the XGBoost model
model = xgb.XGBClassifier(
    use_label_encoder=False, 
    eval_metric='logloss',  
    n_estimators=200,  # Number of trees
    max_depth=5,       # Tree depth
    learning_rate=0.1,  # Step size shrinkage
    subsample=0.8,     # Percentage of samples used per tree
    colsample_bytree=0.8,  # Features per tree
    random_state=42
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Print the evaluation metrics
print(classification_report(y_test, y_pred))