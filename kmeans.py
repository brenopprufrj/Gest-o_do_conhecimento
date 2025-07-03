import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import os
import logging

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Parâmetros para experimentação
K_VALUES = [2, 3, 4, 5, 6, 7, 8, 9, 10]
RANDOM_SEED = 42 # Para reprodutibilidade

def initialize_centroids(X, k):
    indices = np.random.choice(len(X), k, replace=False)
    return X[indices]

def assign_clusters(X, centroids):
    clusters = []
    for point in X:
        distances = [np.linalg.norm(point - centroid) for centroid in centroids]
        cluster = np.argmin(distances)
        clusters.append(cluster)
    return np.array(clusters)

def update_centroids(X, labels, k):
    new_centroids = []
    for i in range(k):
        cluster_points = X[labels == i]
        if len(cluster_points) == 0:
            # Evitar centróide vazio
            new_centroids.append(X[np.random.randint(0, len(X))])
        else:
            new_centroids.append(cluster_points.mean(axis=0))
    return np.array(new_centroids)

def kmeans(X, k, max_iters=100, tol=1e-4):
    centroids = initialize_centroids(X, k)
    for _ in range(max_iters):
        old_centroids = centroids.copy()
        labels = assign_clusters(X, centroids)
        centroids = update_centroids(X, labels, k)
        shift = np.linalg.norm(centroids - old_centroids)
        if shift < tol:
            break
    return labels, centroids

# ----------------
# 🌸 Aplicando no Iris
# ----------------
iris = load_iris()
X_full = iris.data
X_plot = iris.data[:, :2]

import pandas as pd

def main():
    np.random.seed(RANDOM_SEED)
    iris = load_iris()
    X_full = iris.data
    X_plot = iris.data[:, :2]

    os.makedirs("kmeans_results", exist_ok=True)

    results_list = []

    for k in K_VALUES:
        logging.info(f"Running K-means with k={k}")
        labels, centroids = kmeans(X_full, k)
        
        silhouette = 'N/A'
        davies_bouldin = 'N/A'
        calinski_harabasz = 'N/A'

        if k > 1:
            try:
                silhouette = silhouette_score(X_full, labels)
                davies_bouldin = davies_bouldin_score(X_full, labels)
                calinski_harabasz = calinski_harabasz_score(X_full, labels)
                logging.info(f"  -> Silhouette: {silhouette:.3f}, Davies-Bouldin: {davies_bouldin:.3f}, Calinski-Harabasz: {calinski_harabasz:.3f}")
            except Exception as e:
                logging.error(f"  -> Error calculating metrics: {e}")
        else:
            logging.info(f"  -> Metrics not applicable (k=1)")
        
        results_list.append({'k': k,
                             'silhouette_score': silhouette,
                             'davies_bouldin_score': davies_bouldin,
                             'calinski_harabasz_score': calinski_harabasz})
        
        # Plot
        plt.figure()
        for i in range(k):
            plt.scatter(X_plot[labels == i, 0], X_plot[labels == i, 1], label=f"Cluster {i}")
        plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', c='black', s=100, label='Centroids')
        plt.title(f"K-means (k={k})")
        plt.xlabel("Sepal Length")
        plt.ylabel("Sepal Width")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"kmeans_results/k{k}.png")
        plt.close()

    # Salva os resultados em um arquivo CSV
    results_df = pd.DataFrame(results_list)
    results_df.to_csv("kmeans_results/results.csv", index=False)

    logging.info("K-means results saved to kmeans_results/results.csv")

if __name__ == "__main__":
    main()
