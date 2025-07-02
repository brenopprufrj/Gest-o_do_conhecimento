import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score

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

# ... (resto do código permanece o mesmo até a parte da aplicação)

iris = load_iris()
X = iris.data[:, :2]

os.makedirs("dbscan_results", exist_ok=True)

results_list = []

for eps in [0.2, 0.3, 0.4, 0.5]:
    for min_pts in [3, 4, 5]:
        labels = dbscan(X, eps, min_pts)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        score = None  # Valor padrão

        # Silhouette Score
        if n_clusters > 1:
            try:
                score = silhouette_score(X, labels)
                print(f"eps={eps}, min_pts={min_pts} -> Clusters: {n_clusters}, Silhouette Score: {score:.3f}")
            except Exception as e:
                print(f"eps={eps}, min_pts={min_pts} -> Erro no Silhouette: {e}")
        else:
            print(f"eps={eps}, min_pts={min_pts} -> Clusters: {n_clusters}, Silhouette Score: não aplicável")

        results_list.append({
            'eps': eps,
            'min_pts': min_pts,
            'n_clusters': n_clusters,
            'silhouette_score': score if score is not None else 'N/A'
        })

        # Plotagem
        plt.figure()
        for label in set(labels):
            cor = 'k' if label == -1 else plt.cm.tab10(label)
            plt.scatter(X[labels == label, 0], X[labels == label, 1], c=[cor], label=f"Cluster {label}")
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

print("\nResultados do DBSCAN salvos em dbscan_results/results.csv")
