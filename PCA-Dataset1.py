import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

# Read the dataset from a file (assuming it's a CSV file)
file_path = 'datasource/A3-data.txt'
df = pd.read_csv(file_path, delimiter=',', header=0)

# Extract features (X) and target variable (y)
X = df.iloc[:, :-1].values  # All columns except the last one
y = df.iloc[:, -1].values   # The last column

# Standardize the features
X_standardized = StandardScaler().fit_transform(X)

# Perform PCA
pca = PCA()
X_pca = pca.fit_transform(X_standardized)


# Plot the first two principal components
plt.figure(figsize=(12, 5))

# Scatter plot of the first two principal components
plt.subplot(1, 2, 1)
for class_label in set(y):
    plt.scatter(X_pca[y == class_label, 0], X_pca[y == class_label, 1], label=f'Class {class_label}')


print("shape of X_pca", X_pca.shape)
expl = pca.explained_variance_ratio_
print(expl)
print('suma:',sum(expl[0:4]))

plt.title('PCA Projection - First Two Principal Components')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()

# Scree plot with explained variance
plt.subplot(1, 2, 2)
plt.plot(range(1, min(X.shape[1], len(set(y))) + 1), pca.explained_variance_ratio_.cumsum(), marker='o')
plt.title('Scree Plot - Explained Variance')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')

plt.tight_layout()
plt.show()

