import pandas as pd
import matplotlib.pyplot as plt


file_path = 'A3-data.txt'
df = pd.read_csv(file_path, delimiter=',', header=0)

# Display the first few rows of each dataset
print("Dataset head:")
print(df.head())


# Plot data in 2D
fig, ax = plt.subplots()
scatter = ax.scatter(df['x'], df['y'], c=df['class'], cmap='viridis')

# Add labels and a colorbar
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Scatter Plot of the Dataset')
legend = ax.legend(*scatter.legend_elements(), title='Classes')
ax.add_artist(legend)

# Show the plot
plt.show() 

# Plot data in 3d
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(df['x'], df['y'], df['z'], c=df['class'], cmap='viridis')

# Add labels and a colorbar
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('3D Scatter Plot of the Dataset')
legend = ax.legend(*scatter.legend_elements(), title='Classes')
ax.add_artist(legend)

# Show the plot
plt.show() 

# Plot data in 2D
fig, ax = plt.subplots()
scatter = ax.scatter(df['x'], df['z'], c=df['class'], cmap='viridis')

# Add labels and a colorbar
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Scatter Plot of the Dataset')
legend = ax.legend(*scatter.legend_elements(), title='Classes')
ax.add_artist(legend)

# Show the plot
plt.show() 