import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
import seaborn as sns


# Read the dataset from a file (assuming it's a CSV file)
file_path = 'datasource/prnn_fglass-processed.txt'
df = pd.read_csv(file_path, delimiter='\t', header=0)

# Assuming the last column is the class label
X = df.iloc[:, :-1].values
y_real = df.iloc[:, -1].values

# Standardize the data (important for k-means)
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_standardized)

# Set the range of k values to explore
k_values = [2, 3, 4, 5,7,8]

# Visualize k-means results for different k values
for k in k_values:
    # Apply k-means clustering
    kmeans = KMeans(n_clusters=k, random_state=42)
    y_pred = kmeans.fit_predict(X_standardized)

    # Visualize results in a scatter plot
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_pred, palette='viridis', legend='full')
    plt.title(f'k-Means Clustering (k={k})')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.savefig('images/k-means-ds2-k'+str(k)+'.png')
    plt.show()

    # If k equals the real number of classes, use a confusion matrix
    if k == len(set(y_real)):
        confusion_mat = confusion_matrix(y_real, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.savefig('images/k-means-ds2-confusion-matrix-k'+str(k)+'.png')
        plt.show()
