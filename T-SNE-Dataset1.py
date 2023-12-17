import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


# t-SNE with different parameters
perplexities = [10, 30, 50]
tsne_metrics = ['euclidean', 'manhattan', 'chebyshev', 'minkowski', 'cosine']
learning_rate_max=200
learning_rate_step=50

# Read the dataset from a file (assuming it's a txt file with header separated by a comma)
file_path = 'datasource/A3-data.txt'
df = pd.read_csv(file_path, delimiter=',', header=0)

# Extract features (X) and target variable (y)
X = df.iloc[:, :-1].values  # All columns except the last one
y = df.iloc[:, -1].values   # The last column

# Standardize the features
X_standardized = StandardScaler().fit_transform(X)

# t-SNE with different parameters
perplexities = [10, 30, 50]

plt.figure(figsize=(15, 5))

for j, tsne_metric in enumerate(tsne_metrics):
    for learning_rate in range(1, learning_rate_max +1 , learning_rate_step):    
        print('Metric:'+str(tsne_metric) +' Learning rate:'+str(learning_rate))
        plt.figure(figsize=(15, 5))
        for i, perplexity in enumerate(perplexities):
            # Perform t-SNE
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42,metric=tsne_metric,learning_rate=learning_rate)
            X_tsne = tsne.fit_transform(X_standardized)

            # Plot the t-SNE projection
            plt.subplot(1, len(perplexities), i + 1)
            for class_label in set(y):
                plt.scatter(X_tsne[y == class_label, 0], X_tsne[y == class_label, 1], label=f'Class {class_label}')

            plt.title(f't-SNE Metric {tsne_metric} - LR {learning_rate} - Perplexity {perplexity}')
            plt.xlabel('t-SNE Component 1')
            plt.ylabel('t-SNE Component 2')
            plt.legend()

        plt.tight_layout()
        #plt.savefig('t-SNE-{tsne_metric}-{learning_rate}.png')
        plt.savefig('t-SNE-ds1-'+str(tsne_metric)+'-'+str(learning_rate)+'.png')
        #plt.show()
