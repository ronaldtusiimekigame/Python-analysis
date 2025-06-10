import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Load the Iris dataset from sklearn
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Save the dataset to data/iris.csv
import os
os.makedirs('../data', exist_ok=True)
df.to_csv('../data/iris.csv', index=False)

# Basic data exploration
print("Dataset Info:")
print(df.info())
print("\nFirst 5 rows:")
print(df.head())
print("\nSummary Statistics:")
print(df.describe())

# Create plots directory if it doesn't exist
os.makedirs('../plots', exist_ok=True)

# Visualize pair plot
sns.pairplot(df, hue='species')
plt.savefig('../plots/pairplot.png')
plt.close()

# Visualize feature distributions
plt.figure(figsize=(10, 6))
for column in df.columns[:-1]:
    sns.histplot(data=df, x=column, hue='species', multiple='stack')
    plt.title(f'Distribution of {column}')
    plt.savefig(f'../plots/{column}_hist.png')
    plt.close()

print("EDA completed. Visualizations saved in the 'plots' folder.")