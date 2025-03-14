# Import necessary libraries
from ucimlrepo import fetch_ucirepo
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Fetch the Bone Marrow Transplantation for Children dataset
bone_marrow_transplant_children = fetch_ucirepo(id=565)

# Extract features and targets as pandas DataFrames
X = bone_marrow_transplant_children.data.features
y = bone_marrow_transplant_children.data.targets

# Handle non-numeric columns by applying Label Encoding
for column in X.columns:
    if X[column].dtype == 'object':  # Check if the column is non-numeric
        X[column] = LabelEncoder().fit_transform(X[column])  # Apply LabelEncoder

# Calculate the correlation matrix
correlation_matrix = X.corr()

# Create the heatmap with adjusted figure size and font size
plt.figure(figsize=(12, 10))  # I
ncrease figure size for better visibility
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5,
            annot_kws={'size': 8},  # Adjust font size of annotations (make it smaller)
            cbar_kws={'shrink': 0.8})  # Adjust color bar size

# Rotate x and y axis labels for better readability
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=45, va='top')

# Set title and show the plot
plt.title('Correlation Heatmap')
plt.tight_layout()  # Adjust layout to prevent overlap
plt.show()
