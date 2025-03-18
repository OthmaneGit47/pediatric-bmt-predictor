import unittest
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from unittest.mock import patch, MagicMock

class TestSHAPAnalysis(unittest.TestCase):
    
    def setUp(self):
        """Set up test environment before each test method"""
        # Create a small synthetic dataset for testing
        np.random.seed(42)
        n_samples = 100
        n_features = 5
        
        # Create synthetic data
        self.X_synthetic = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        self.y_synthetic = np.random.randint(0, 2, size=n_samples)
        
        # Path to the actual dataset (adjust if needed)
        self.dataset_path = 'data/Processed_data.csv'
        
    def test_data_loading(self):
        """Test if the dataset can be loaded properly"""
        # Skip this test if the file doesn't exist
        if not os.path.exists(self.dataset_path):
            self.skipTest(f"Dataset file {self.dataset_path} not found")
            
        # Try loading the dataset
        df = pd.read_csv(self.dataset_path)
        
        # Check if 'survival_status' column exists
        self.assertIn('survival_status', df.columns, 
                      "Dataset must contain 'survival_status' column")
        
        # Check if there's enough data
        self.assertGreater(len(df), 10, "Dataset should have more than 10 rows")
        
    def test_train_test_split(self):
        """Test the train-test split functionality"""
        # Perform the split
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_synthetic, self.y_synthetic, test_size=0.2, random_state=42
        )
        
        # Check split sizes
        expected_train_size = int(0.8 * len(self.X_synthetic))
        expected_test_size = len(self.X_synthetic) - expected_train_size
        
        self.assertEqual(len(X_train), expected_train_size, 
                         "Training set size doesn't match expected size")
        self.assertEqual(len(X_test), expected_test_size, 
                         "Test set size doesn't match expected size")
    
    def test_model_training(self):
        """Test if model trains successfully and produces reasonable results"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_synthetic, self.y_synthetic, test_size=0.2, random_state=42
        )
        
        # Train model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate accuracy (just check that it runs, not the actual value)
        accuracy = accuracy_score(y_test, y_pred)
        self.assertIsInstance(accuracy, float, "Accuracy should be a float")
        self.assertGreaterEqual(accuracy, 0.0, "Accuracy should be >= 0")
        self.assertLessEqual(accuracy, 1.0, "Accuracy should be <= 1")
    
    def test_shap_explainer_creation(self):
        """Test if SHAP explainer can be created successfully"""
        # Train a small model
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_synthetic, self.y_synthetic, test_size=0.2, random_state=42
        )
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Create SHAP explainer
        explainer = shap.TreeExplainer(model)
        
        # Check explainer type
        self.assertIsInstance(explainer, shap.explainers.Tree, 
                             "Explainer should be a TreeExplainer")
    
    def test_shap_values_calculation(self):
        """Test if SHAP values can be calculated correctly"""
        # Train a small model
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_synthetic, self.y_synthetic, test_size=0.2, random_state=42
        )
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Create SHAP explainer and calculate values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        
        # Check if shap_values has the correct structure
        if isinstance(shap_values, list):
            # Multi-class case
            self.assertEqual(len(shap_values), 2, 
                            "For binary classification, should have 2 sets of SHAP values")
            self.assertEqual(shap_values[0].shape[0], X_test.shape[0], 
                            "Number of samples in SHAP values should match X_test")
            self.assertEqual(shap_values[0].shape[1], X_test.shape[1], 
                            "Number of features in SHAP values should match X_test")
        else:
            # Single-class case
            self.assertEqual(shap_values.shape[0], X_test.shape[0], 
                            "Number of samples in SHAP values should match X_test")
            self.assertEqual(shap_values.shape[1], X_test.shape[1], 
                            "Number of features in SHAP values should match X_test")
    
    def test_shap_processing(self):
        """Test SHAP values processing logic"""
        # Set up test data
        X_test = pd.DataFrame(np.random.randn(20, 5), 
                             columns=[f'feature_{i}' for i in range(5)])
        
        # Case 1: Multi-class (list of arrays)
        shap_values_list = [
            np.random.randn(20, 5),  # Class 0
            np.random.randn(20, 5)   # Class 1
        ]
        
        # Process as in the original code
        if isinstance(shap_values_list, list):
            processed_values = np.mean(np.abs(shap_values_list), axis=0)
            
        # Check result
        self.assertEqual(processed_values.shape, (20, 5), 
                        "Processed values should maintain sample and feature dimensions")
        
        # Case 2: 3D array
        shap_values_3d = np.random.randn(20, 5, 2)  # 20 samples, 5 features, 2 outputs
        
        # Process as in the original code
        if shap_values_3d.ndim == 3:
            processed_values = shap_values_3d.mean(axis=2)
            
        # Check result
        self.assertEqual(processed_values.shape, (20, 5), 
                        "Processed 3D values should reduce to 2D")
    
    @patch('matplotlib.pyplot.show')
    @patch('shap.summary_plot')
    def test_visualization(self, mock_summary_plot, mock_show):
        """Test the visualization components"""
        # Setup
        X_test = pd.DataFrame(np.random.randn(20, 5), 
                             columns=[f'feature_{i}' for i in range(5)])
        shap_values = np.random.randn(20, 5)
        
        # Create importance DataFrame
        shap_importance = pd.DataFrame({
            "Feature": X_test.columns,
            "SHAP Importance": np.abs(shap_values).mean(axis=0)
        }).sort_values(by="SHAP Importance", ascending=False)
        
        # Test SHAP summary plot call
        shap.summary_plot(shap_values, X_test)
        mock_summary_plot.assert_called_once()
        
        # Test bar plot creation
        plt.figure(figsize=(10, 6))
        plt.barh(shap_importance["Feature"], shap_importance["SHAP Importance"])
        plt.xlabel("Mean |SHAP Value|")
        plt.ylabel("Feature")
        plt.title("Feature Importance via SHAP")
        plt.gca().invert_yaxis()
        plt.show()
        
        mock_show.assert_called_once()
    
    def test_integration(self):
        """Perform an integration test of the entire workflow"""
        # Skip if the dataset doesn't exist
        if not os.path.exists(self.dataset_path):
            self.skipTest(f"Dataset file {self.dataset_path} not found")
        
        try:
            # Load data
            df = pd.read_csv(self.dataset_path)
            X = df.drop('survival_status', axis=1)
            y = df['survival_status'].values.ravel()
            
            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train model
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            model.fit(X_train, y_train)
            
            # SHAP explanation
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test.iloc[:5])  # Use only 5 samples for speed
            
            # Process SHAP values
            if isinstance(shap_values, list):
                shap_values = np.mean(np.abs(shap_values), axis=0)
                
            shap_values = np.array(shap_values)
            if shap_values.ndim == 3:
                shap_values = shap_values.mean(axis=2)
                
            # Verify feature count
            self.assertEqual(shap_values.shape[1], X_test.shape[1], 
                            "Feature count mismatch!")
            
            # Create importance DataFrame
            shap_importance = pd.DataFrame({
                "Feature": X.columns,
                "SHAP Importance": np.abs(shap_values).mean(axis=0)
            }).sort_values(by="SHAP Importance", ascending=False)
            
            # Check results
            self.assertEqual(len(shap_importance), len(X.columns), 
                            "Should have importance for all features")
            self.assertGreater(shap_importance["SHAP Importance"].sum(), 0, 
                              "Sum of importance values should be positive")
            
        except Exception as e:
            self.fail(f"Integration test failed with error: {str(e)}")

if __name__ == '__main__':
    unittest.main()