Load and Explore the Dataset

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load the Iris dataset from sklearn
iris = load_iris()

# Convert the dataset into a pandas DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Add the target variable (species) as a column
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# Display the first few rows to inspect the data
print(df.head())

# Explore the structure of the dataset
print("\nData Types:")
print(df.dtypes)

# Check for any missing values
print("\nMissing Values:")
print(df.isnull().sum())

Output of Step 1

   sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm) species
0                5.1               3.5                1.4               0.2  setosa
1                4.9               3.0                1.4               0.2  setosa
2                4.7               3.2                1.3               0.2  setosa
3                4.6               3.1                1.5               0.2  setosa
4                5.0               3.6                1.4               0.2  setosa

Data Types:
sepal length (cm)     float64
sepal width (cm)      float64
petal length (cm)     float64
petal width (cm)      float64
species             category
dtype: object

Missing Values:
sepal length (cm)    0
sepal width (cm)     0
petal length (cm)    0
petal width (cm)     0
species              0
dtype: int64

Basic Data Analysis
Summary Statistics: Use .describe() to get the basic statistics of the numerical columns.

# Compute basic statistics of numerical columns
print(df.describe())

# Group by the 'species' column and compute the mean for each group
grouped_by_species = df.groupby('species').mean()
print("\nMean of numerical columns by species:")
print(grouped_by_species)

Output of Step 2

       sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
count            150.000000         150.000000          150.000000         150.000000
mean               5.843333           3.057333            3.758000           1.199333
std                0.828066           0.435866            1.765298           0.762238
min                4.300000           2.000000            1.000000           0.100000
25%                5.100000           2.800000            1.600000           0.300000
50%                5.800000           3.000000            4.350000           1.300000
75%                6.400000           3.300000            5.100000           1.800000
max                7.900000           4.400000            6.900000           2.500000

Mean of numerical columns by species:
                  sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
species                                                                      
setosa                     5.006            3.418               1.464               0.244
versicolor                 5.936            2.770               4.260               1.326
virginica                  6.588            2.974               5.552               2.026


Data Visualization
Line Chart: Trend over time (e.g., sales, if the dataset had a time series)
Since the Iris dataset does not have a time variable, let's assume we're visualizing the trend of one feature (e.g., sepal length) across species. However, I'll add a synthetic time column for demonstration.

# Create a synthetic time column
df['time'] = pd.date_range(start='2023-01-01', periods=len(df), freq='D')

# Line plot: Plotting sepal length over time (just for trial)
plt.figure(figsize=(10, 6))
sns.lineplot(x='time', y='sepal length (cm)', data=df)
plt.title('Sepal Length Over Time')
plt.xlabel('Time')
plt.ylabel('Sepal Length (cm)')
plt.xticks(rotation=45)
plt.show()

Bar Chart: Comparison of average petal length by species

# Bar plot: Average petal length by species
plt.figure(figsize=(10, 6))
sns.barplot(x='species', y='petal length (cm)', data=df)
plt.title('Average Petal Length by Species')
plt.xlabel('Species')
plt.ylabel('Average Petal Length (cm)')
plt.show()

Histogram: Distribution of sepal width

# Histogram: Distribution of sepal width
plt.figure(figsize=(10, 6))
sns.histplot(df['sepal width (cm)'], bins=20, kde=True)
plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.show()

Scatter Plot: Relationship between sepal length and petal length

# Scatter plot: Sepal length vs Petal length
plt.figure(figsize=(10, 6))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df)
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.show()

Summary of Findings
Data Structure: The Iris dataset has 150 rows and 5 columns, with no missing values.
Basic Statistics: The dataset provides descriptive statistics like mean, standard deviation, and percentiles. Sepal length has a mean of about 5.84 cm, while petal length has a mean of 3.76 cm.
Grouping by Species: The species exhibit clear differences in the mean values for all features. For example, setosa has the smallest mean petal length and petal width, whereas virginica has the largest.
Visualizations:
Bar Chart shows that setosa has the smallest petal lengths.
Scatter Plot shows a positive relationship between sepal length and petal length.
Histogram demonstrates that the distribution of sepal width is slightly left-skewed.
