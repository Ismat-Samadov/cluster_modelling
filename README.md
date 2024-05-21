Cluster Modelling

This repository contains a comprehensive approach to clustering analysis using K-Means and Agglomerative Clustering on a dataset of customer transactions. The project involves data preprocessing, model optimization using Optuna, and result visualization.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Optimization](#optimization)
- [Evaluation](#evaluation)
- [Results](#results)

## Overview

This project aims to perform clustering analysis on customer transaction data to identify distinct groups of customers based on their spending behavior. The project involves the following steps:
1. Data Preprocessing
2. Clustering using K-Means and Agglomerative Clustering
3. Optimization of the number of clusters using Optuna
4. Evaluation and comparison of clustering models
5. Visualization of clustering results

## Dataset

The dataset used in this project is `cluster 2.csv`. It contains customer transaction data with various features like balance, purchases, cash advance, credit limit, payments, etc.

## Requirements

To run this project, you need the following libraries:

- pandas
- numpy
- scikit-learn
- seaborn
- matplotlib
- optuna

You can install the required libraries using the following command:

```bash
pip install -r requirements.txt
```

## Usage

To run the clustering analysis, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/Ismat-Samadov/cluster_modelling.git
    ```
2. Navigate to the project directory:
    ```bash
    cd cluster_modelling
    ```
3. Ensure you have the necessary libraries installed:
    ```bash
    pip install -r requirements.txt
    ```
4. Run the Jupyter notebook:
    ```bash
    jupyter notebook Cluster_Modelling.ipynb
    ```

## Project Structure

The repository contains the following files:

- `Cluster_Modelling.ipynb`: Jupyter notebook containing the clustering analysis and optimization code.
- `cluster 2.csv`: Dataset used for clustering analysis.
- `requirements.txt`: List of required libraries.

## Optimization

We optimize the number of clusters for both K-Means and Agglomerative Clustering using Optuna. The optimization aims to maximize the silhouette score, which measures how similar an object is to its own cluster compared to other clusters.

The optimization functions are defined as follows:

```python
def optimize_kmeans(trial):
    n_clusters = trial.suggest_int('n_clusters', 2, 10)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(numeric_df)
    return silhouette_score(numeric_df, kmeans_labels)

def optimize_agg(trial):
    n_clusters = trial.suggest_int('n_clusters', 2, 10)
    agg = AgglomerativeClustering(n_clusters=n_clusters)
    agg_labels = agg.fit_predict(numeric_df)
    return silhouette_score(numeric_df, agg_labels)

# Optimize K-Means
kmeans_study = optuna.create_study(direction='maximize')
kmeans_study.optimize(optimize_kmeans, n_trials=20)
best_kmeans_clusters = kmeans_study.best_params['n_clusters']

# Optimize Agglomerative Clustering
agg_study = optuna.create_study(direction='maximize')
agg_study.optimize(optimize_agg, n_trials=20)
best_agg_clusters = agg_study.best_params['n_clusters']
```

## Evaluation

We evaluate the models using the silhouette score. The evaluation results are compiled into a DataFrame for easy comparison:

```python
results = pd.DataFrame({
    'Model': ['K-Means', 'Agglomerative'],
    'Default Silhouette Score': [
        silhouette_score(scaled_df, kmeans_labels),
        silhouette_score(scaled_df, agg_labels)
    ],
    'Optimized Silhouette Score': [
        kmeans_study.best_value,
        agg_study.best_value
    ]
})
```

## Results

The optimized number of clusters and the corresponding silhouette scores are printed:

```python
print(f"Best number of clusters for K-Means: {best_kmeans_clusters}")
print(f"Best number of clusters for Agglomerative Clustering: {best_agg_clusters}")
```

The clustering results are then visualized using pair plots:

```python
sns.pairplot(cluster_data, hue='Best_KMeans_Labels')
plt.show()
```

## Contributing

If you would like to contribute to this project, please create a pull request with detailed information about the changes.

## Contact

For any questions or inquiries, please contact [Ismat Samadov](https://github.com/Ismat-Samadov).