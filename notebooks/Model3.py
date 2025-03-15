import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd


df = pd.read_csv('data\processed_data_v3.csv')






X = df.drop('survival_status', axis=1)  # All columns except 'survival_status'
y = df['survival_status']  # The target column

# Split dataset into training & testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)

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