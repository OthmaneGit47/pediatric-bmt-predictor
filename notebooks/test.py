import pytest
import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

@pytest.fixture
def sample_data():
    """Load and prepare sample dataset for testing."""
    df = pd.read_csv("data/processed_data_v3.csv")
    
    # Extract features and target
    X = df.drop(columns=["survival_status"])
    y = df["survival_status"]
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test

def test_model_training(sample_data):
    """Test if the LightGBM model trains correctly and produces predictions."""
    X_train, X_test, y_train, y_test = sample_data
    
    # Initialize model
    model = lgb.LGBMClassifier(
        objective='binary',
        boosting_type='gbdt',
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        n_estimators=100
    )

    # Train model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Check that the model produces predictions
    assert len(y_pred) == len(y_test), "Number of predictions does not match test set size"
    
    # Ensure predictions are in expected range (0 or 1)
    assert set(y_pred).issubset({0, 1}), "Predictions contain unexpected values"
    
    # Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    assert accuracy > 0.5, f"Model accuracy too low: {accuracy}"  # Should be better than random guessing
sample_data = sample_data()
















































































































































































































































































