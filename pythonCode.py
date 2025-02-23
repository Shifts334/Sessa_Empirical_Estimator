import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import ecdf
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo

# Step 1: Fetch the Iris dataset using ucimlrepo
iris = fetch_ucirepo(id=53)

# Data (as pandas dataframes)
X = iris.data.features  # Features (e.g., sepal length, sepal width, petal length, petal width)
y = iris.data.targets   # Targets (e.g., type of iris)

# Metadata and variable information
print(iris.metadata)
print(iris.variables)

# Step 2: Data Preparation
# For the purpose of this example, we will treat the Iris dataset as a prescription dataset.
# We will assume that each row represents a prescription, and the features represent attributes of the prescription.
# We will add a 'fill_date' column to simulate prescription data.

# Create a synthetic 'fill_date' column
np.random.seed(1234)
start_date = pd.to_datetime('2023-01-01')
end_date = pd.to_datetime('2023-12-31')
X['fill_date'] = pd.to_datetime(np.random.choice(pd.date_range(start_date, end_date), size=len(X)))

# Add a 'patient_id' column to simulate multiple patients
X['patient_id'] = np.random.randint(1, 10, size=len(X))  # Assume 9 patients

# Sort the data by patient_id and fill_date
X = X.sort_values(by=['patient_id', 'fill_date'])

# Step 3: Compute ECDF of temporal distances
X['prev_fill_date'] = X.groupby('patient_id')['fill_date'].shift(1)
X['temporal_distance'] = (X['fill_date'] - X['prev_fill_date']).dt.days
ecdf_func = ecdf(X['temporal_distance'].dropna())
x = np.sort(X['temporal_distance'].dropna())
y = ecdf_func(x)

# Retain only 80% of the ECDF
dfper = pd.DataFrame({'x': x, 'y': y})
dfper = dfper[dfper['y'] <= 0.8]

# Step 4: Randomly select a pair of consecutive prescriptions
random_pairs = X.groupby('patient_id').apply(lambda x: x.sample(1)).reset_index(drop=True)
random_pairs = random_pairs[['patient_id', 'fill_date', 'prev_fill_date', 'temporal_distance']]

# Step 5: Standardization and Clustering
temporal_distances = random_pairs['temporal_distance'].values.reshape(-1, 1)
temporal_distances = (temporal_distances - np.mean(temporal_distances)) / np.std(temporal_distances)

# Determine optimal number of clusters using Silhouette Analysis
silhouette_scores = []
for n_clusters in range(2, 11):
    kmeans = KMeans(n_clusters=n_clusters, random_state=1234)
    cluster_labels = kmeans.fit_predict(temporal_distances)
    silhouette_avg = silhouette_score(temporal_distances, cluster_labels)
    silhouette_scores.append(silhouette_avg)

optimal_clusters = np.argmax(silhouette_scores) + 2  # +2 because range starts at 2

# Perform K-means clustering with optimal number of clusters
kmeans = KMeans(n_clusters=optimal_clusters, random_state=1234)
random_pairs['cluster'] = kmeans.fit_predict(temporal_distances)

# Step 6: Build PDF and find median temporal distance for each cluster
cluster_summary = random_pairs.groupby('cluster')['temporal_distance'].agg(['min', 'max', 'median']).reset_index()
cluster_summary['median'] = np.exp(cluster_summary['median'])  # Convert back to original scale

# Step 7: Compute end of supply for each filled prescription
random_pairs = pd.merge(random_pairs, cluster_summary[['cluster', 'median']], on='cluster', how='left')
random_pairs['end_of_supply'] = random_pairs['fill_date'] + pd.to_timedelta(random_pairs['median'], unit='d')

# Output the results
print("Results of the SEE method applied to the Iris dataset:")
print(random_pairs)
print("\nCluster Summary:")
print(cluster_summary)