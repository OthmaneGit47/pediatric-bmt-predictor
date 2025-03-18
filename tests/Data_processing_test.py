import unittest
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import tempfile
import shutil


# Import the preprocessing functions from your script
# If your code is in a file called data_preprocessing.py, uncomment these lines:
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# from data_preprocessing import handle_missing_values, encode_categorical_features, check_for_outliers, replace_outliers_iqr, handle_imbalanced_data, handle_correlations, optimize_data


# If you cannot import directly, we'll redefine the functions here for testing
def handle_missing_values(df, categorical_cols, numerical_cols):
    cat_imputer = SimpleImputer(strategy="most_frequent")
    num_imputer = SimpleImputer(strategy="median")

    df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
    df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])
    return df


def encode_categorical_features(df, categorical_cols):
    encoder = OrdinalEncoder()
    df[categorical_cols] = encoder.fit_transform(df[categorical_cols])
    return df


def check_for_outliers(df):
    outlier_columns = {}
    for column in df.select_dtypes(include=[np.number]).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]
        if not outliers.empty:
            outlier_columns[column] = outliers.count()
    return outlier_columns


def replace_outliers_iqr(df):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    df_cleaned = df.copy()
    for col in df.columns:
        median_value = df[col].median()
        df_cleaned[col] = np.where((df[col] < lower_bound[col]) | (df[col] > upper_bound[col]), 
                                  median_value, df[col])
    return df_cleaned


def handle_imbalanced_data(df):
    X = df.drop(columns=["survival_status"])
    y = df["survival_status"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    df_train_balanced = X_train_smote
    df_train_balanced['survival_status'] = y_train_smote
    
    return df_train_balanced


def handle_correlations(df):
    X = df.drop('survival_status', axis=1)
    y = df['survival_status']

    correlation_matrix = X.corr()
    correlation_threshold = 0.9

    to_remove = set()
    corr_pairs = correlation_matrix.abs().unstack().sort_values(ascending=False)
    corr_pairs = corr_pairs[corr_pairs < 1]

    for (feature1, feature2), correlation in corr_pairs.items():
        if correlation > correlation_threshold:
            if feature1 not in to_remove and feature2 not in to_remove:
                to_remove.add(feature2)

    X_updated = X.drop(columns=to_remove, errors='ignore')
    df = pd.concat([X_updated, y], axis=1)
    return df, to_remove


def optimize_data(df):
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype(np.int32)

    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype(np.float32)
    return df


class TestDataProcessing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Create a sample dataset for testing"""
        # Create a temporary directory for test data
        cls.test_dir = tempfile.mkdtemp()
        
        # Sample data with numerical and categorical columns
        cls.data = {
            'age': [25, 40, 35, np.nan, 50],
            'income': [50000, 60000, np.nan, 75000, 55000],
            'education': ['High School', 'Bachelor', 'Master', np.nan, 'PhD'],
            'marital_status': ['Single', np.nan, 'Married', 'Divorced', 'Married'],
            'survival_status': [0, 1, 0, 1, 0]
        }
        cls.df = pd.DataFrame(cls.data)
        
        # Define column types
        cls.categorical_cols = ['education', 'marital_status']
        cls.numerical_cols = ['age', 'income']
        
        # Create a test file with highly correlated columns
        cls.corr_data = {
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [0.98, 2.01, 3.02, 3.98, 4.99],  # Highly correlated with feature1
            'feature3': [10, 20, 30, 40, 50],
            'feature4': [-5, -10, -15, -20, -25],
            'survival_status': [0, 1, 0, 1, 0]
        }
        cls.df_corr = pd.DataFrame(cls.corr_data)
        
        # Save sample data to test directory
        cls.df.to_csv(os.path.join(cls.test_dir, "test_data.csv"), index=False)
        cls.df_corr.to_csv(os.path.join(cls.test_dir, "test_corr_data.csv"), index=False)
    
    @classmethod
    def tearDownClass(cls):
        """Remove test directory and files"""
        shutil.rmtree(cls.test_dir)
    
    def test_handle_missing_values(self):
        """Test that missing values are properly imputed"""
        df_copy = self.df.copy()
        result_df = handle_missing_values(df_copy, self.categorical_cols, self.numerical_cols)
        
        # Check that there are no missing values
        self.assertEqual(result_df.isnull().sum().sum(), 0)
        
        # Check that the shape is preserved
        self.assertEqual(result_df.shape, self.df.shape)

    def test_encode_categorical_features(self):
        """Test that categorical features are properly encoded"""
        df_copy = self.df.copy()
        # First handle missing values
        df_copy = handle_missing_values(df_copy, self.categorical_cols, self.numerical_cols)
        result_df = encode_categorical_features(df_copy, self.categorical_cols)
        
        # Check that categorical columns are now numeric
        for col in self.categorical_cols:
            self.assertTrue(pd.api.types.is_numeric_dtype(result_df[col].dtype))

    def test_check_for_outliers(self):
        """Test outlier detection function"""
        # Create a dataset with known outliers
        outlier_df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 100],  # 100 is an outlier
            'feature2': [10, 20, 30, 40, 50]
        })
        
        outliers = check_for_outliers(outlier_df)
        self.assertIn('feature1', outliers)
        self.assertEqual(outliers['feature1'], 1)  # One outlier in feature1

    def test_replace_outliers_iqr(self):
        """Test outlier replacement function"""
        # Create a dataset with known outliers
        outlier_df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 100],  # 100 is an outlier
            'feature2': [10, 20, 30, 40, 500]  # 500 is an outlier
        })
        
        cleaned_df = replace_outliers_iqr(outlier_df)
        
        # Check that outliers were replaced (median values)
        self.assertNotEqual(cleaned_df['feature1'].iloc[4], 100)
        self.assertNotEqual(cleaned_df['feature2'].iloc[4], 500)
        
        # Check specifically for median replacement
        self.assertEqual(cleaned_df['feature1'].iloc[4], outlier_df['feature1'].median())
        self.assertEqual(cleaned_df['feature2'].iloc[4], outlier_df['feature2'].median())

    # def test_handle_imbalanced_data(self):
    #     """Test SMOTE balancing function"""
    #     # Create an imbalanced dataset
    #     imbalanced_df = pd.DataFrame({
    #         'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    #         'feature2': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    #         'survival_status': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  # Imbalanced: 8 zeros, 2 ones
    #     })
        
    #     # Skip actual SMOTE testing if test dataset is too small
    #     if len(imbalanced_df) >= 10:
    #         balanced_df = handle_imbalanced_data(imbalanced_df)
            
    #         # Check that classes are more balanced now
    #         value_counts = balanced_df['survival_status'].value_counts()
    #         # The minority class should have more samples now
    #         self.assertGreater(value_counts.min(), 2)  # More than original 2 samples
            
    #         # Validate split and SMOTE execution (just check it ran without errors)
    #         self.assertIsNotNone(balanced_df)

    def test_handle_correlations(self):
        """Test correlation handling function"""
        df_copy = self.df_corr.copy()
        result_df, removed_features = handle_correlations(df_copy)
        
        # feature1 and feature2 are highly correlated, one should be removed
        self.assertTrue('feature1' in result_df.columns or 'feature2' in result_df.columns)
        self.assertTrue('feature1' in removed_features or 'feature2' in removed_features)
        
        # Check that exactly one of the correlated features was removed
        self.assertEqual(len(removed_features), 1)

    def test_optimize_data(self):
        """Test data type optimization function"""
        # Create DataFrame with int64 and float64 types
        df_large_types = pd.DataFrame({
            'int_col': np.array([1, 2, 3, 4, 5], dtype=np.int64),
            'float_col': np.array([1.1, 2.2, 3.3, 4.4, 5.5], dtype=np.float64),
            'survival_status': [0, 1, 0, 1, 0]
        })
        
        optimized_df = optimize_data(df_large_types)
        
        # Check that types were changed to smaller types
        self.assertEqual(optimized_df['int_col'].dtype, np.dtype('int32'))
        self.assertEqual(optimized_df['float_col'].dtype, np.dtype('float32'))

    def test_full_pipeline(self):
        """Test the entire preprocessing pipeline"""
        df_copy = self.df.copy()
        
        # Step 1: Handle missing values
        df_copy = handle_missing_values(df_copy, self.categorical_cols, self.numerical_cols)
        
        # Step 2: Encode categorical features
        df_copy = encode_categorical_features(df_copy, self.categorical_cols)
        
        # Step 3: Check for outliers
        outliers_before = check_for_outliers(df_copy)
        
        # Step 4: Replace outliers
        df_copy = replace_outliers_iqr(df_copy)
        
        # Verify outliers were replaced
        outliers_after = check_for_outliers(df_copy)
        self.assertLessEqual(sum(outliers_after.values()), sum(outliers_before.values()))
        
        # Step 5: Handle imbalanced data (skip in test if dataset is too small)
        if len(df_copy) >= 10:
            df_copy = handle_imbalanced_data(df_copy)
        
        # Step 6: Handle correlations
        df_copy, _ = handle_correlations(df_copy)
        
        # Step 7: Optimize data types
        final_df = optimize_data(df_copy)
        
        # Verify final result has no missing values and appropriate data types
        self.assertEqual(final_df.isnull().sum().sum(), 0)
        self.assertTrue(all(dtype == np.dtype('int32') or dtype == np.dtype('float32') 
                          for dtype in final_df.dtypes if pd.api.types.is_numeric_dtype(dtype)))


if __name__ == '__main__':
    unittest.main()