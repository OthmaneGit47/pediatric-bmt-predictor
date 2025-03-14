import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import LabelEncoder

# Fetch dataset from UCI repository
bone_marrow_transplant_children = fetch_ucirepo(id=565)

# Extract features (X) and target variable (y)
X = bone_marrow_transplant_children.data.features
y = bone_marrow_transplant_children.data.targets

# Combine features and target into a single DataFrame for easier processing
df = pd.concat([X, y], axis=1)



# Display basic info about the dataset
print(df.info())

# Display first few rows
print(df.head())


# Count missing values for each column
missing_counts = df.isnull().sum()
print(missing_counts)   

# Calculate missing percentage
missing_percentage = (missing_counts / len(df)) * 100

# Display only columns with missing values
missing_data = pd.DataFrame({"Missing Count": missing_counts, "Missing %": missing_percentage})
missing_data = missing_data[missing_data["Missing Count"] > 0]
print(missing_data) #The maximum percentage of missing values is 16% therefore we can just replace them with new values

# Adress categorical features and replace missing values
numerical = df.select_dtypes(include=["number"]).columns
for col in numerical:
    unique_values = df[col].nunique()
    if unique_values > 2 and unique_values < 10:
        df[col] = df[col].fillna(df[col].mode()[0])
        df[col] = df[col].astype(str)

# Identify categorical features
categorical_cols = df.select_dtypes(include=["object"]).columns


# Identify numerical columns
numerical_cols = df.select_dtypes(include=["number"]).columns


# Fill missing values with the median for numerical columns
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())


# Transform categorical features into numerical
df = pd.get_dummies(df, drop_first=False)
print(df.info())

for col in df.select_dtypes(include=['int64']).columns:
    df[col] = df[col].astype(np.int32)

for col in df.select_dtypes(include=['float64']).columns:
    df[col] = df[col].astype(np.float32)
print(df.info())

