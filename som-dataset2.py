import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score
from minisom import MiniSom
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
file_path = 'datasource/prnn_fglass-processed.txt'
df = pd.read_csv(file_path, delimiter='\t', header=0)

# Last column is the class label
X = df.iloc[:, :-1].values
y_real = df.iloc[:, -1].values

# Standardize the data
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)

neighborhood_functions=['bubble','gaussian']
learning_rates=[0.5,1,10]
topologies=['rectangular','hexagonal']
sizes=[10,12] # for more than 100 neurons

best_neighborhood_function='gaussian'
best_learning_rate=0.5
best_topology='rectangular'
best_size=10
best_score=0

for neighborhood_function in neighborhood_functions:
    for learning_rate in learning_rates:
        for topology in topologies:
            for size in sizes:
                map_size = (size, size) 
                # Create SOM
                som = MiniSom(map_size[0], map_size[1], X_standardized.shape[1],learning_rate=learning_rate,
                            neighborhood_function=neighborhood_function,topology=topology)
                som.random_weights_init(X_standardized)
                # Train the SOM with a specific number of iterations
                som.train_random(X_standardized, 500)

                # Heatmap of the most represented class in each position
                plt.figure(figsize=(10, 8))
                winners = np.array([som.winner(x) for x in X_standardized])
                mapped_labels = np.array([y_real[win] for win in winners])

                # Heatmap of the most represented class
                sns.heatmap(mapped_labels, cmap='autumn', cbar=True)
                plt.title('SOM Heatmap of Most Represented Class')
                plt.xlabel('SOM Columns')
                plt.ylabel('SOM Rows')
                plt.savefig('images/som/som-ds2-'+neighborhood_function+'-'+topology+'-'+str(learning_rate)+'-'+str(size)+'-heatmap.png')
                #plt.show()

                #Adjusted Rand index
                winners_flat = winners[:, 0] * map_size[1] + winners[:, 1]
                ari = adjusted_rand_score(y_real, winners_flat)
                print(f'Adjusted Rand Index: {ari * 100:.2f}% ' + neighborhood_function+'-'+topology+'-'+str(learning_rate)+'-'+str(size))
                if(ari>best_score):
                    best_neighborhood_function=neighborhood_function
                    best_learning_rate=learning_rate
                    best_size=size
                    best_topology=topology

                # U-matrix
                umatrix = som.distance_map()

                plt.figure(figsize=(10, 8))
                sns.heatmap(umatrix, cmap='autumn', cbar=True)
                plt.title('SOM U-Matrix')
                plt.xlabel('SOM Columns')
                plt.ylabel('SOM Rows')
                plt.savefig('images/som/som-ds2-'+neighborhood_function+'-'+topology+'-'+str(learning_rate)+'-'+str(size)+'-u-matrix.png')
                #plt.show()

 # For the best score
print('Best plot', best_size,best_learning_rate,best_neighborhood_function,best_topology)
best_map_size = (best_size, best_size)

best_som = MiniSom(best_map_size[0], best_map_size[1], X_standardized.shape[1],
                learning_rate=best_learning_rate,
                neighborhood_function=best_neighborhood_function,topology=best_topology)

best_som.random_weights_init(X_standardized)
best_som.train_random(X_standardized, 500)

# Plot component
plt.figure(figsize=(15, 12))
for i, feature in enumerate(df.columns[:-1]):
    plt.subplot(3, 3, i + 1)
    plt.title(f'Component Plane: {feature}')
    plt.pcolor(best_som.get_weights()[:,:,i], cmap='autumn')
    plt.colorbar()

plt.tight_layout()
plt.savefig('images/som/som-ds2-best-dendogram'+best_neighborhood_function+'-'+best_topology+'-'+str(best_learning_rate)+'-'+str(best_size)+'.png')
#plt.show()
