import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Load the cleaned dataset
df = pd.read_csv("processed_data_v1.csv")


# Display the first few rows
print(df.info())

# Check class distribution
print("\nClass Distribution:")
print(df["survival_status"].value_counts())

# Calculate percentage of each class
print("\nClass Percentage:")
print(df["survival_status"].value_counts(normalize=True) * 100)


# Split features and survival_status
X = df.drop(columns=["survival_status"])  # Features
y = df["survival_status"]                 # survival_status variable

# Split into train & test sets before applying SMOTE (to prevent data leakage)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE only on the training data
smote = SMOTE(sampling_strategy="auto", random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Check new class distribution after SMOTE
print("\nClass Distribution After SMOTE:")
print(y_train_resampled.value_counts())

sns.countplot(x=y_train_resampled)
plt.title("Class Distribution After SMOTE")
plt.show()

resampled_df = pd.concat([X_train_resampled, y_train_resampled], axis=1)
resampled_df.to_csv("processed_data_v2.csv", index=False)

