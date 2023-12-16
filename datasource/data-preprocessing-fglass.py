# read th information of a CSV and load into a dataframe
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# read the csv file
name='prnn_fglass'

df= pd.read_csv(name+'.csv',sep=',')
print(df.head())

print(df.describe())

df_processed = pd.DataFrame()

# Replace 'unknown' with the mode in each column
for column in df.columns:
    mode_value = df[column].mode()[0]
    df_processed[column] = df[column].replace('unknown', mode_value)
    is_yes_no = df_processed[column].isin(['window_float_glass', 'window_non-float_glass','vehicle_glass','containers','tableware','vehicle_headlamp_glass']).all()
    if is_yes_no:
        df_processed[column] = df_processed[column].replace({'window_float_glass': 0, 'window_non-float_glass': 1,'vehicle_glass':2,'containers':3,'tableware':4,'vehicle_headlamp_glass':5}).astype(int)
    else:
        is_numeric = pd.to_numeric(df_processed[column], errors='coerce').notnull().all()
        if not is_numeric:
            unique=df_processed[column].unique()
            unique={k:v for v,k in enumerate(unique)}
            print(unique)
            df_processed[column]=df_processed[column].map(unique)

        print(column, is_numeric,is_yes_no)

#remove id column
df_processed = df_processed.drop(df_processed.columns[0], axis=1)
df_processed = df_processed.drop(df_processed.columns[7], axis=1)
df_processed = df_processed.drop(df_processed.columns[7], axis=1)

# Data normalization
# Min-Max Scaling
df_processed.iloc[:, 0]= (df_processed.iloc[:,0] - df_processed.iloc[:,0].min()) / (df_processed.iloc[:,0].max() - df_processed.iloc[:,0].min())
df_processed.iloc[:, 1]= (df_processed.iloc[:,1] - df_processed.iloc[:,1].min()) / (df_processed.iloc[:,1].max() - df_processed.iloc[:,1].min())
df_processed.iloc[:, 2]= (df_processed.iloc[:,2] - df_processed.iloc[:,2].min()) / (df_processed.iloc[:,2].max() - df_processed.iloc[:,2].min())
df_processed.iloc[:, 3]= (df_processed.iloc[:,3] - df_processed.iloc[:,3].min()) / (df_processed.iloc[:,3].max() - df_processed.iloc[:,3].min())
df_processed.iloc[:, 4]= (df_processed.iloc[:,4] - df_processed.iloc[:,4].min()) / (df_processed.iloc[:,4].max() - df_processed.iloc[:,4].min())
df_processed.iloc[:, 5]= (df_processed.iloc[:,5] - df_processed.iloc[:,5].min()) / (df_processed.iloc[:,5].max() - df_processed.iloc[:,5].min())
df_processed.iloc[:, 6]= (df_processed.iloc[:,6] - df_processed.iloc[:,6].min()) / (df_processed.iloc[:,6].max() - df_processed.iloc[:,6].min())
#df_processed.iloc[:, 7]= (df_processed.iloc[:,7] - df_processed.iloc[:,7].min()) / (df_processed.iloc[:,7].max() - df_processed.iloc[:,7].min())
#df_processed.iloc[:, 8]= (df_processed.iloc[:,8] - df_processed.iloc[:,8].min()) / (df_processed.iloc[:,8].max() - df_processed.iloc[:,8].min())


df_processed = df_processed.sample(frac=1, random_state=42)

print(df_processed.head())

output_file_name=name+'-processed.txt'
df_processed.to_csv(output_file_name,sep='\t', index=False,header=None)

# Plot data in 2D
fig, ax = plt.subplots()
scatter = ax.scatter(df_processed['RI'], df_processed['Na'], c=df_processed['type'], cmap='viridis')

# Add labels and a colorbar
ax.set_xlabel('RI')
ax.set_ylabel('Na')
ax.set_title('Scatter Plot of the Dataset')
legend = ax.legend(*scatter.legend_elements(), title='Classes')
ax.add_artist(legend)

# Show the plot
plt.show() 


# Plot data in 3d
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(df_processed['RI'], df_processed['Na'], df_processed['Mg'], c=df_processed['type'], cmap='viridis')

# Add labels and a colorbar
ax.set_xlabel('RI')
ax.set_ylabel('Na')
ax.set_zlabel('Mg')
ax.set_title('3D Scatter Plot of the Dataset')
legend = ax.legend(*scatter.legend_elements(), title='Classes')
ax.add_artist(legend)

# Show the plot
plt.show() 

# Plot data in 2D
fig, ax = plt.subplots()
scatter = ax.scatter(df_processed['Al'], df_processed['Si'], c=df_processed['type'], cmap='viridis')

# Add labels and a colorbar
ax.set_xlabel('Al')
ax.set_ylabel('Si')
ax.set_title('Scatter Plot of the Dataset')
legend = ax.legend(*scatter.legend_elements(), title='Classes')
ax.add_artist(legend)

# Show the plot
plt.show() 


# Plot data in 3d
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(df_processed['Al'], df_processed['Si'], df_processed['K'], c=df_processed['type'], cmap='viridis')

# Add labels and a colorbar
ax.set_xlabel('Al')
ax.set_ylabel('Si')
ax.set_zlabel('K')
ax.set_title('3D Scatter Plot of the Dataset')
legend = ax.legend(*scatter.legend_elements(), title='Classes')
ax.add_artist(legend)

# Show the plot
plt.show() 