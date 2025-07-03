import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Parâmetros para experimentação
EPS_VALUES = [0.2, 0.3, 0.4, 0.5]
MIN_PTS_VALUES = [3, 4, 5]
RANDOM_SEED = 42 # Para reprodutibilidade

def euclidean(p1, p2):
    return np.linalg.norm(p1 - p2)

def region_query(X, point_idx, eps):
    neighbors = []
    for i in range(len(X)):
        if euclidean(X[point_idx], X[i]) <= eps:
            neighbors.append(i)
    return neighbors

def expand_cluster(X, labels, point_idx, neighbors, cluster_id, eps, min_pts):
    labels[point_idx] = cluster_id
    i = 0
    while i < len(neighbors):
        neighbor_idx = neighbors[i]
        if labels[neighbor_idx] == -1:  # Noise
            labels[neighbor_idx] = cluster_id
        elif labels[neighbor_idx] == 0:
            labels[neighbor_idx] = cluster_id
            new_neighbors = region_query(X, neighbor_idx, eps)
            if len(new_neighbors) >= min_pts:
                neighbors += new_neighbors
        i += 1

def dbscan(X, eps, min_pts):
    labels = [0] * len(X)  # 0 = unvisited, -1 = noise, >0 = cluster
    cluster_id = 0
    for i in range(len(X)):
        if labels[i] != 0:
            continue
        neighbors = region_query(X, i, eps)
        if len(neighbors) < min_pts:
            labels[i] = -1  # noise
        else:
            cluster_id += 1
            expand_cluster(X, labels, i, neighbors, cluster_id, eps, min_pts)
    return np.array(labels)


import pandas as pd

def main():
    np.random.seed(RANDOM_SEED)
    iris = load_iris()
    X_full = iris.data
    X_plot = iris.data[:, :2]

    os.makedirs("dbscan_results", exist_ok=True)

    results_list = []

    for eps in EPS_VALUES:
        for min_pts in MIN_PTS_VALUES:
            logging.info(f"Running DBSCAN with eps={eps}, min_pts={min_pts}")
            labels = dbscan(X_full, eps, min_pts)
            
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
            silhouette = 'N/A'
            davies_bouldin = 'N/A'
            calinski_harabasz = 'N/A'

            if n_clusters > 1:
                try:
                    silhouette = silhouette_score(X_full, labels)
                    davies_bouldin = davies_bouldin_score(X_full, labels)
                    calinski_harabasz = calinski_harabasz_score(X_full, labels)
                    logging.info(f"  -> Clusters: {n_clusters}, Silhouette: {silhouette:.3f}, Davies-Bouldin: {davies_bouldin:.3f}, Calinski-Harabasz: {calinski_harabasz:.3f}")
                except Exception as e:
                    logging.error(f"  -> Error calculating metrics: {e}")
            else:
                logging.info(f"  -> Clusters: {n_clusters}, Metrics not applicable (1 cluster or all noise)")

            results_list.append({
                'eps': eps,
                'min_pts': min_pts,
                'n_clusters': n_clusters,
                'silhouette_score': silhouette,
                'davies_bouldin_score': davies_bouldin,
                'calinski_harabasz_score': calinski_harabasz
            })

            # Plotagem
            plt.figure()
            for label in set(labels):
                cor = 'k' if label == -1 else plt.cm.tab10(label)
                plt.scatter(X_plot[labels == label, 0], X_plot[labels == label, 1], c=[cor], label=f"Cluster {label}")
            plt.title(f"DBSCAN - eps={eps}, min_pts={min_pts}")
            plt.xlabel("Sepal Length")
            plt.ylabel("Sepal Width")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"dbscan_results/eps{eps}_minpts{min_pts}.png")
            plt.close()

    # Salva os resultados em um arquivo CSV
    results_df = pd.DataFrame(results_list)
    results_df.to_csv("dbscan_results/results.csv", index=False)

    logging.info("DBSCAN results saved to dbscan_results/results.csv")

if __name__ == "__main__":
    main()
