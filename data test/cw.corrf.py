# Import necessary libraries
from ucimlrepo import fetch_ucirepo
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Fetch the Bone Marrow Transplantation for Children dataset
bone_marrow_transplant_children = fetch_ucirepo(id=565)

# Extract features and targets as pandas DataFrames
X = bone_marrow_transplant_children.data.features.copy()
y = bone_marrow_transplant_children.data.targets.copy()

# Handle non-numeric columns by applying Label Encoding
label_encoder = LabelEncoder()
for column in X.select_dtypes(include=['object']).columns:
    X[column] = label_encoder.fit_transform(X[column])

# Calculate the correlation matrix
correlation_matrix = X.corr()

# Define a threshold for high correlation
correlation_threshold = 0.85

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
X_updated.to_csv("updated_dataset.csv", index=False)

# Display the new correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(
    X_updated.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5,
    annot_kws={'size': 8}, cbar_kws={'shrink': 0.8}
)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=45, va='top')
plt.title('Correlation Heatmap (After Feature Selection)')
plt.tight_layout()
plt.show()

# Show the top remaining correlated feature pairs
remaining_corr_pairs = X_updated.corr().abs().unstack().sort_values(ascending=False)
remaining_corr_pairs = remaining_corr_pairs[remaining_corr_pairs < 1]  # Exclude self-correlation

top_remaining_corr = remaining_corr_pairs[remaining_corr_pairs > 0.5]  # Adjust threshold as needed
print("\nTop Correlating Feature Pairs After Selection:")
print(top_remaining_corr)
