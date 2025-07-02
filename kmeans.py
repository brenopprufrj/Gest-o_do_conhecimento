import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score
import os

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
X = iris.data[:, :2]

import pandas as pd

# ... (resto do código permanece o mesmo até a parte da aplicação)

# ----------------
# 🌸 Aplicando no Iris
# ----------------
iris = load_iris()
X = iris.data[:, :2]

os.makedirs("kmeans_results", exist_ok=True)

results_list = []

for k in [2, 3, 4, 5, 6]:
    labels, centroids = kmeans(X, k)
    
    score = -1.0  # Valor padrão
    if k > 1:
        try:
            score = silhouette_score(X, labels)
            print(f"k={k} -> Silhouette Score: {score:.3f}")
        except Exception as e:
            print(f"k={k} -> Erro no Silhouette: {e}")
    
    results_list.append({'k': k, 'silhouette_score': score})
    
    # Plot
    plt.figure()
    for i in range(k):
        plt.scatter(X[labels == i, 0], X[labels == i, 1], label=f"Cluster {i}")
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

print("\nResultados do K-means salvos em kmeans_results/results.csv")
