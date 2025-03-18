import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import OrdinalEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import os

df = pd.read_csv("data\Raw_data.csv")
print(df.info())

# Separate the columns into categorical and numerical
categorical_cols = [col for col in df.columns if df[col].dtype == "object"]
numerical_cols = [col for col in df.columns if df[col].dtype in ["int64", "float64"]]


print((df.isnull().sum() / len(df)) * 100) #The maximum amount of missing values in the dataset is 16% therefore we can just replace them with appropriate values
def handle_missing_values(df):
    # Impute missing values
    cat_imputer = SimpleImputer(strategy="most_frequent")  # Mode for categorical
    num_imputer = SimpleImputer(strategy="median")  # Median for numerical

    df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
    df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])

handle_missing_values(df)
print(df.isnull().sum())
print(df["extcGvHD"])

def encode_categorical_features(df):
    encoder = OrdinalEncoder()
    df[categorical_cols] = encoder.fit_transform(df[categorical_cols])

print(df["Disease"])
encode_categorical_features(df)
print(df["Disease"])

def check_for_outliers(df):
    outlier_columns = {}  # Dictionary to store outlier counts for each feature

    for column in df.select_dtypes(include=[np.number]).columns:  # Only numerical columns
        Q1 = df[column].quantile(0.25)  # 25th percentile (lower quartile)
        Q3 = df[column].quantile(0.75)  # 75th percentile (upper quartile)
        IQR = Q3 - Q1  # Interquartile Range

        lower_bound = Q1 - 1.5 * IQR  # Define the lower bound for normal values
        upper_bound = Q3 + 1.5 * IQR  # Define the upper bound for normal values

        # Find outliers (values outside the lower and upper bounds)
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)][column]

        if not outliers.empty:  # If there are outliers in this column
            outlier_columns[column] = outliers.count()  # Store the count of outliers

    return outlier_columns  # Return a dictionary with column names and outlier counts
print(check_for_outliers(df))

def replace_outliers_iqr(df):
    Q1 = df.quantile(0.25)  # Premier quartile
    Q3 = df.quantile(0.75)  # Troisième quartile
    IQR = Q3 - Q1           # Intervalle interquartile
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Remplacer les valeurs inférieures au seuil inférieur par la médiane
    df_cleaned = df.copy()
    for col in df.columns:
        median_value = df[col].median()
        df_cleaned[col] = np.where((df[col] < lower_bound[col]) | (df[col] > upper_bound[col]), 
                                   median_value, df[col])
    return df_cleaned

df = replace_outliers_iqr(df)


def handle_imbalanced_data(df):
    # Split first (to avoid data leakage)
    X = df.drop(columns=["survival_status"])
    y = df["survival_status"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Apply SMOTE only on training data to handle the imbalance
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    # Return the new balanced training dataset along with the original testing dataset
    # (test data remains unchanged as we don't want to apply SMOTE on it)
    df_train_balanced = X_train_smote
    df_train_balanced['survival_status'] = y_train_smote
    
    # Return the entire dataset (balanced train, original test)
    return df_train_balanced

df = handle_imbalanced_data(df)
print(df["survival_status"].value_counts())

def handle_correlations(df):
    X = df.drop('survival_status', axis=1)  # All columns except 'survival_status'
    y = df['survival_status']  # The target column


    # Calculate the correlation matrix
    correlation_matrix = X.corr()

    # Define a threshold for high correlation
    correlation_threshold = 0.9

    # Identify highly correlated feature pairs
    to_remove = set()
    corr_pairs = correlation_matrix.abs().unstack().sort_values(ascending=False)
    corr_pairs = corr_pairs[corr_pairs < 1]  # Remove self-correlation

    # Select one feature to remove from each correlated pair
    for (feature1, feature2), correlation in corr_pairs.items():
        if correlation > correlation_threshold:
            if feature1 not in to_remove and feature2 not in to_remove:
                to_remove.add(feature2)  # Keep feature1, remove feature2

    # Drop the selected features
    X_updated = X.drop(columns=to_remove)

    # Display number of dropped features
    print(f"Removed {len(to_remove)} correlated features: {to_remove}")

    # Save the updated dataset
    df = pd.concat([X_updated, y], axis=1)
    return df

df = handle_correlations(df)

def optimize_data(df):
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = df[col].astype(np.int32)

    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = df[col].astype(np.float32)
    return df

print(df.info())
df = optimize_data(df)
print(df.info())


if os.path.exists("data\Processed_data.csv") == False:
    df.to_csv("data\Processed_data.csv",index=False)
else:
    print("The file already exists")








