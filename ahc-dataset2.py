import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from scipy.cluster.hierarchy import dendrogram, linkage
import seaborn as sns

# Load dataset
file_path = 'datasource/prnn_fglass-processed.txt'
df = pd.read_csv(file_path, delimiter='\t', header=0)

# Assuming the last column is the class label
X = df.iloc[:, :-1].values
y_real = df.iloc[:, -1].values

# Standardize the data
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

# Calculate Euclidean distances
distances = pairwise_distances(X_standardized, metric='euclidean')

# Apply Agglomerative Hierarchical Clustering with UPGMA linkage
linkage_matrix_upgma = linkage(distances, method='average')

# Plot dendrogram for UPGMA
plt.figure(figsize=(12, 8))
dendrogram(linkage_matrix_upgma, labels=y_real, color_threshold=1.5, leaf_rotation=90, leaf_font_size=10)
plt.title('UPGMA Dendrogram')
plt.xlabel('Patterns')
plt.ylabel('Euclidean Distance')
plt.savefig('images/ahc/ahc-ds2-upgma-dendogram.png')
plt.show()

# Apply Agglomerative Hierarchical Clustering with Complete linkage
linkage_matrix_complete = linkage(distances, method='complete')

# Plot dendrogram for Complete linkage
plt.figure(figsize=(12, 8))
dendrogram(linkage_matrix_complete, labels=y_real, color_threshold=10, leaf_rotation=90, leaf_font_size=10)
plt.title('Complete Linkage Dendrogram')
plt.xlabel('Patterns')
plt.ylabel('Euclidean Distance')
plt.savefig('images/ahc/ahc-ds2-complete-dendogram.png')
plt.show()
