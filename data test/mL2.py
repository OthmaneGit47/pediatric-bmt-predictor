import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Step 1: Load your dataset (adjust this according to your dataset file)
# For example, assume the dataset is in a CSV file called 'pediatric_bone_marrow.csv'
df = pd.read_csv('cleaned_bone_marrow_dataset.csv')

# Step 2: Preprocess the data (handle missing values, encode categorical features, etc.)
# Let's assume that your target column is 'survival_status' and the rest are features
# Adjust the following lines based on the dataset structure

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()



# Step 3: Separate features (X) and target variable (y)
X = df.drop('survival_status', axis=1)  # All columns except 'survival_status'
y = df['survival_status']  # The target column

# Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Define the SVM model
model = SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced')

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Print the evaluation metrics
print(classification_report(y_test, y_pred))
