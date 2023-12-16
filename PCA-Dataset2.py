import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

# Read the dataset from a file (assuming it's a CSV file)
file_path = 'datasource/prnn_fglass-processed.txt'
df = pd.read_csv(file_path, delimiter='\t', header=0)

# Extract features (X) and target variable (y)
X = df.iloc[:, :-1].values  # All columns except the last one
y = df.iloc[:, -1].values   # The last column

# Standardize the features
X_standardized = StandardScaler().fit_transform(X)

# Perform PCA
pca = PCA()
X_pca = pca.fit_transform(X_standardized)


# Plot the first two principal components
plt.figure(figsize=(7, 5))

# Scatter plot of the first two principal components
plt.subplot(1, 1, 1)
for class_label in set(y):
    plt.scatter(X_pca[y == class_label, 0], X_pca[y == class_label, 1], label=f'Class {class_label}')


print("shape of X_pca", X_pca.shape)
expl = pca.explained_variance_ratio_
print(expl)
print('suma:',sum(expl[0:X_pca.shape[1]]))

plt.title('PCA Projection - First Two Principal Components')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()

# Scree plot with explained variance
plt.tight_layout()
plt.show()

#graficamos el acumulado de varianza explicada en las nuevas dimensiones
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()