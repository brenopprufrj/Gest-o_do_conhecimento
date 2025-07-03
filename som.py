import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import os
import logging

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Parâmetros para experimentação
GRID_SHAPES = [(2, 2), (3, 3), (4, 4), (5, 5), (6, 6)]
SIGMA_VALUES = [0.2, 0.5, 1.0, 1.5, 2.0]
LEARNING_RATES = [0.01, 0.1, 0.5, 0.9, 1.0]
EPOCHS_VALUES = [50, 100, 200, 300, 500]
RANDOM_SEED = 42 # Para reprodutibilidade

def som_train(X, grid_shape=(3, 3), epochs=100, sigma=1.0, learning_rate=0.5):
    h, w = grid_shape
    n_neurons = h * w
    input_dim = X.shape[1]

    # Inicializa pesos aleatórios dos neurônios
    weights = np.random.rand(n_neurons, input_dim)

    # Posiciona neurônios em grade 2D
    neuron_coords = np.array([[i, j] for i in range(h) for j in range(w)])

    def neighborhood_function(dist, sigma):
        return np.exp(-dist ** 2 / (2 * sigma ** 2))

    sigma_initial = sigma
    lr_initial = learning_rate

    for epoch in range(epochs):
        # Decaimento linear dos parâmetros
        current_lr = lr_initial * (1 - epoch / epochs)
        current_sigma = sigma_initial * (1 - epoch / epochs)

        for x in X:
            # Encontra o neurônio mais próximo (BMU)
            dists = np.linalg.norm(weights - x, axis=1)
            bmu_idx = np.argmin(dists)
            bmu_coord = neuron_coords[bmu_idx]

            # Atualiza cada neurônio com base na distância ao BMU
            for i in range(n_neurons):
                dist_to_bmu = np.linalg.norm(neuron_coords[i] - bmu_coord)
                influence = neighborhood_function(dist_to_bmu, current_sigma)
                weights[i] += current_lr * influence * (x - weights[i])

    return weights, neuron_coords

def som_assign(X, weights):
    labels = []
    for x in X:
        dists = np.linalg.norm(weights - x, axis=1)
        labels.append(np.argmin(dists))
    return np.array(labels)

# ----------------
# 🌸 Aplicando no Iris
# ----------------
iris = load_iris()
X_full = iris.data
X_plot = iris.data[:, :2]

os.makedirs("som_results", exist_ok=True)

import pandas as pd

def main():
    np.random.seed(RANDOM_SEED)
    iris = load_iris()
    X_full = iris.data
    X_plot = iris.data[:, :2]

    os.makedirs("som_results", exist_ok=True)

    results_list = []

    for grid in GRID_SHAPES:
        for sigma_initial in SIGMA_VALUES:
            for lr_initial in LEARNING_RATES:
                for epochs_num in EPOCHS_VALUES:
                    logging.info(f"Training SOM: grid={grid}, sigma={sigma_initial}, lr={lr_initial}, epochs={epochs_num}")
                    
                    weights, coords = som_train(
                        X_full,
                        grid_shape=grid,
                        epochs=epochs_num,
                        sigma=sigma_initial,
                        learning_rate=lr_initial
                    )
                    labels = som_assign(X_full, weights)

                    # --- Análise e Plotagem ---
                    unique_labels = np.unique(labels)
                    n_clusters = len(unique_labels)
                    
                    cluster_counts = {label: np.sum(labels == label) for label in unique_labels}
                    logging.info(f"  -> Cluster counts: {cluster_counts}")

                    silhouette = 'N/A'
                    davies_bouldin = 'N/A'
                    calinski_harabasz = 'N/A'

                    if n_clusters > 1:
                        try:
                            silhouette = silhouette_score(X_full, labels)
                            davies_bouldin = davies_bouldin_score(X_full, labels)
                            calinski_harabasz = calinski_harabasz_score(X_full, labels)
                            logging.info(f"  -> Silhouette: {silhouette:.3f}, Davies-Bouldin: {davies_bouldin:.3f}, Calinski-Harabasz: {calinski_harabasz:.3f}")
                        except Exception as e:
                            logging.error(f"  -> Error calculating metrics: {e}")
                            silhouette = 'Error'
                            davies_bouldin = 'Error'
                            calinski_harabasz = 'Error'
                    else:
                        logging.info("  -> Metrics not applicable (1 cluster or all noise)")
                        silhouette = 'N/A'
                        davies_bouldin = 'N/A'
                        calinski_harabasz = 'N/A'
                    
                    results_list.append({
                        'grid_shape': f"{grid[0]}x{grid[1]}",
                        'sigma': sigma_initial,
                        'learning_rate': lr_initial,
                        'epochs': epochs_num,
                        'n_clusters': n_clusters,
                        'cluster_counts': str(cluster_counts), # Store as string for CSV
                        'silhouette_score': silhouette,
                        'davies_bouldin_score': davies_bouldin,
                        'calinski_harabasz_score': calinski_harabasz
                    })

                    # Plot
                    plt.figure(figsize=(8, 6))
                    for i in sorted(list(set(labels))):
                        plt.scatter(X_plot[labels == i, 0], X_plot[labels == i, 1], label=f"Cluster {i}")
                    
                    title_text = f"SOM {grid[0]}x{grid[1]} | σ={sigma_initial}, LR={lr_initial}, Epochs={epochs_num}"
                    if isinstance(silhouette, float):
                        title_text += f"\nSilhouette: {silhouette:.3f}"
                    if isinstance(davies_bouldin, float):
                        title_text += f", DB: {davies_bouldin:.3f}"
                    if isinstance(calinski_harabasz, float):
                        title_text += f", CH: {calinski_harabasz:.3f}"

                    plt.title(title_text)
                    plt.xlabel("Sepal Length")
                    plt.ylabel("Sepal Width")
                    plt.legend()
                    plt.tight_layout()
                    
                    # Salva a figura com nome descritivo
                    filename = f"som_results/som_{grid[0]}x{grid[1]}_sigma{sigma_initial}_lr{lr_initial}_e{epochs_num}.png"
                    plt.savefig(filename)
                    plt.close()

    # Salva os resultados em um arquivo CSV
    results_df = pd.DataFrame(results_list)
    results_df.to_csv("som_results/results.csv", index=False)

    logging.info("SOM results saved to som_results/results.csv")

if __name__ == "__main__":
    main()

