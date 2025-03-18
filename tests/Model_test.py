import unittest
import numpy as np
import pandas as pd
import os
import joblib
from unittest.mock import patch, MagicMock
import tempfile
import random


class TestModelPrediction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test environment with mock model and sample data"""
        # Create a temporary directory for testing
        cls.test_dir = tempfile.mkdtemp()
        
        # Define the features used by the model
        cls.FEATURES = [
            'Recipientgender', 'Stemcellsource', 'Donorage', 'Donorage35', 'IIIV', 
            'Gendermatch', 'DonorABO', 'RecipientABO', 'RecipientRh', 'ABOmatch', 
            'CMVstatus', 'DonorCMV', 'RecipientCMV', 'Disease', 'Riskgroup', 
            'Txpostrelapse', 'Diseasegroup', 'HLAmatch', 'HLAmismatch', 'Antigen', 
            'Allele', 'Recipientage10', 'Recipientageint', 'Relapse', 'aGvHDIIIIV', 
            'extcGvHD', 'CD34kgx10d6', 'CD3dCD34', 'CD3dkgx10d8', 'Rbodymass', 
            'ANCrecovery', 'PLTrecovery', 'time_to_aGvHD_III_IV', 'survival_time'
        ]
        
        # Sample test data
        cls.sample_data = {
            'Recipientgender': 1.0,
            'Stemcellsource': 1.0,
            'Donorage': 21.128767,
            'Donorage35': 0.0,
            'IIIV': 1.0,
            'Gendermatch': 0.0,
            'DonorABO': 0.0,
            'RecipientABO': 0.0,
            'RecipientRh': 1.0,
            'ABOmatch': 0.0,
            'CMVstatus': 0.0,
            'DonorCMV': 0.0,
            'RecipientCMV': 0.0,
            'Disease': 0.0,
            'Riskgroup': 1.0,
            'Txpostrelapse': 0.0,
            'Diseasegroup': 1.0,
            'HLAmatch': 0.0,
            'HLAmismatch': 0.0,
            'Antigen': -1.0,
            'Allele': -1.0,
            'Recipientage10': 1.0,
            'Recipientageint': 2.0,
            'Relapse': 0.0,
            'aGvHDIIIIV': 1.0,
            'extcGvHD': 1.0,
            'CD34kgx10d6': 5.08,
            'CD3dCD34': 0.709805,
            'CD3dkgx10d8': 7.16,
            'Rbodymass': 54.1,
            'ANCrecovery': 15.0,
            'PLTrecovery': 17.0,
            'time_to_aGvHD_III_IV': 1000000.0,
            'survival_time': 20.0
        }
        
        # Create a mock model file
        cls.create_mock_model()
    
    @classmethod
    def tearDownClass(cls):
        """Clean up the test environment"""
        # Remove the temporary test directory
        import shutil
        shutil.rmtree(cls.test_dir)
    
    @classmethod
    def create_mock_model(cls):
        """Create a mock RandomForest model file for testing"""
        try:
            # Try to import scikit-learn for creating a real but simple model
            from sklearn.ensemble import RandomForestClassifier
            
            # Create a simple model
            model = RandomForestClassifier(n_estimators=2, random_state=42)
            
            # Generate some random training data
            X_train = np.random.rand(10, len(cls.FEATURES))
            y_train = np.random.randint(0, 2, 10)
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Save the model
            cls.model_path = os.path.join(cls.test_dir, 'test_model.pkl')
            joblib.dump(model, cls.model_path)
        
        except ImportError:
            # If scikit-learn is not available, create a mock model object
            mock_model = MagicMock()
            mock_model.predict.return_value = np.array([0])  # Default prediction: Live
            
            # Save the mock model
            cls.model_path = os.path.join(cls.test_dir, 'test_model.pkl')
            joblib.dump(mock_model, cls.model_path)
    
    def test_model_loading(self):
        """Test that the model can be loaded correctly"""
        model = joblib.load(self.model_path)
        self.assertIsNotNone(model)
    
    def test_feature_ordering(self):
        """Test that feature ordering is correct"""
        self.assertEqual(len(self.FEATURES), 34)
        self.assertEqual(self.FEATURES[0], 'Recipientgender')
        self.assertEqual(self.FEATURES[-1], 'survival_time')
    
    def test_input_data_preparation(self):
        """Test the preparation of input data"""
        # Convert patient data to the format expected by the model
        input_data = np.array([self.sample_data[feature] for feature in self.FEATURES]).reshape(1, -1)
        
        # Check shape and content
        self.assertEqual(input_data.shape, (1, len(self.FEATURES)))
        self.assertEqual(input_data[0, 0], 1.0)  # First feature (Recipientgender)
        self.assertEqual(input_data[0, -1], 20.0)  # Last feature (survival_time)
    
    def test_prediction_output(self):
        """Test the prediction function with the model"""
        # Load the model
        model = joblib.load(self.model_path)
        
        # Prepare input data
        input_data = np.array([self.sample_data[feature] for feature in self.FEATURES]).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(input_data)
        
        # Check prediction format
        self.assertIn(prediction[0], [0, 1])
    
    @patch('builtins.print')
    def test_prediction_interpretation(self, mock_print):
        """Test the interpretation of prediction results"""
        # Load the model
        model = joblib.load(self.model_path)
        
        # Case 1: Survival (0)
        with patch.object(model, 'predict', return_value=np.array([0])):
            # Prepare input data
            input_data = np.array([self.sample_data[feature] for feature in self.FEATURES]).reshape(1, -1)
            
            # Make prediction and interpret
            prediction = model.predict(input_data)
            
            if prediction[0] == 0:
                print("Prediction: Live (Success)")
            else:
                print("Prediction: Die (Failure)")
            
            # Check that the correct message was printed
            mock_print.assert_called_with("Prediction: Live (Success)")
        
        # Case 2: Death (1)
        mock_print.reset_mock()
        with patch.object(model, 'predict', return_value=np.array([1])):
            # Prepare input data
            input_data = np.array([self.sample_data[feature] for feature in self.FEATURES]).reshape(1, -1)
            
            # Make prediction and interpret
            prediction = model.predict(input_data)
            
            if prediction[0] == 0:
                print("Prediction: Live (Success)")
            else:
                print("Prediction: Die (Failure)")
            
            # Check that the correct message was printed
            mock_print.assert_called_with("Prediction: Die (Failure)")
    
    def test_with_random_data(self):
        """Test the model with multiple random patient data samples"""
        # Load the model
        model = joblib.load(self.model_path)
        
        # Generate 5 random patient data samples
        for _ in range(5):
            random_patient = self.generate_random_patient_data()
            
            # Prepare input data
            input_data = np.array([random_patient[feature] for feature in self.FEATURES]).reshape(1, -1)
            
            # Make prediction
            prediction = model.predict(input_data)
            
            # Verify prediction is valid
            self.assertIn(prediction[0], [0, 1])
    
    def generate_random_patient_data(self):
        """Generate random patient data within realistic bounds"""
        patient_data = {}
        
        # Binary features (0 or 1)
        binary_features = [
            'Recipientgender', 'Stemcellsource', 'Donorage35', 'IIIV', 'Gendermatch', 
            'DonorABO', 'RecipientABO', 'RecipientRh', 'ABOmatch', 'CMVstatus', 
            'DonorCMV', 'RecipientCMV', 'Disease', 'Riskgroup', 'Txpostrelapse', 
            'Diseasegroup', 'HLAmatch', 'HLAmismatch', 'Relapse', 'aGvHDIIIIV', 'extcGvHD'
        ]
        
        for feature in binary_features:
            patient_data[feature] = float(random.randint(0, 1))
        
        # Special categorical features
        patient_data['Antigen'] = float(random.choice([-1, 0, 1]))
        patient_data['Allele'] = float(random.choice([-1, 0, 1]))
        patient_data['Recipientage10'] = float(random.randint(0, 5))
        patient_data['Recipientageint'] = float(random.randint(0, 5))
        
        # Continuous features with realistic ranges
        patient_data['Donorage'] = float(random.uniform(18, 60))
        patient_data['CD34kgx10d6'] = float(random.uniform(1, 10))
        patient_data['CD3dCD34'] = float(random.uniform(0.1, 2))
        patient_data['CD3dkgx10d8'] = float(random.uniform(1, 15))
        patient_data['Rbodymass'] = float(random.uniform(40, 100))
        patient_data['ANCrecovery'] = float(random.uniform(10, 30))
        patient_data['PLTrecovery'] = float(random.uniform(10, 30))
        patient_data['time_to_aGvHD_III_IV'] = float(random.choice([random.uniform(10, 100), 1000000]))
        patient_data['survival_time'] = float(random.uniform(1, 60))
        
        return patient_data
    
    def test_prediction_with_pandas_dataframe(self):
        """Test making predictions using a pandas DataFrame"""
        try:
            # Skip if pandas not available
            import pandas as pd
            
            # Load the model
            model = joblib.load(self.model_path)
            
            # Create a pandas DataFrame with the sample data
            df = pd.DataFrame([self.sample_data])
            
            # Make sure columns are in the correct order
            df = df[self.FEATURES]
            
            # Make prediction
            prediction = model.predict(df)
            
            # Verify prediction is valid
            self.assertIn(prediction[0], [0, 1])
            
        except ImportError:
            # Skip test if pandas is not available
            self.skipTest("Pandas not available")


# Helper function to run prediction on a single patient
def predict_survival(patient_data, model_path='randomforest_model1.pkl', features=None):
    """
    Predicts survival for a patient using the loaded model.
    
    Args:
        patient_data (dict): Dictionary containing patient features
        model_path (str): Path to the saved model file
        features (list): List of feature names in the correct order
        
    Returns:
        int: 0 for survival, 1 for death
        str: Human-readable prediction result
    """
    # Load the model
    model = joblib.load(model_path)
    
    # Use default features if none provided
    if features is None:
        features = [
            'Recipientgender', 'Stemcellsource', 'Donorage', 'Donorage35', 'IIIV', 
            'Gendermatch', 'DonorABO', 'RecipientABO', 'RecipientRh', 'ABOmatch', 
            'CMVstatus', 'DonorCMV', 'RecipientCMV', 'Disease', 'Riskgroup', 
            'Txpostrelapse', 'Diseasegroup', 'HLAmatch', 'HLAmismatch', 'Antigen', 
            'Allele', 'Recipientage10', 'Recipientageint', 'Relapse', 'aGvHDIIIIV', 
            'extcGvHD', 'CD34kgx10d6', 'CD3dCD34', 'CD3dkgx10d8', 'Rbodymass', 
            'ANCrecovery', 'PLTrecovery', 'time_to_aGvHD_III_IV', 'survival_time'
        ]
    
    # Convert the patient data into a numpy array
    input_data = np.array([patient_data[feature] for feature in features]).reshape(1, -1)
    
    # Make a prediction
    prediction = model.predict(input_data)
    
    # Return result
    result_text = "Live (Success)" if prediction[0] == 0 else "Die (Failure)"
    return prediction[0], result_text


if __name__ == '__main__':
    unittest.main()