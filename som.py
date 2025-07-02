import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score
import os

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

    for epoch in range(epochs):
        for x in X:
            # Encontra o neurônio mais próximo (BMU)
            dists = np.linalg.norm(weights - x, axis=1)
            bmu_idx = np.argmin(dists)
            bmu_coord = neuron_coords[bmu_idx]

            # Atualiza cada neurônio com base na distância ao BMU
            for i in range(n_neurons):
                dist_to_bmu = np.linalg.norm(neuron_coords[i] - bmu_coord)
                influence = neighborhood_function(dist_to_bmu, sigma)
                weights[i] += learning_rate * influence * (x - weights[i])

        # Decaimento de parâmetros
        sigma *= 0.9
        learning_rate *= 0.9

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
X = iris.data[:, :2]

os.makedirs("som_results", exist_ok=True)

import pandas as pd

# ... (resto do código permanece o mesmo até a parte da aplicação)

# ----------------
# 🌸 Aplicando no Iris
# ----------------
iris = load_iris()
X = iris.data[:, :2]

os.makedirs("som_results", exist_ok=True)

results_list = []

for grid in [(2, 2), (3, 3), (4, 4)]:
    for sigma_initial in [0.5, 1.0, 1.5]:
        for lr_initial in [0.1, 0.5, 0.9]:
            for epochs_num in [50, 100, 200]:
                print(f"Training SOM: grid={grid}, sigma={sigma_initial}, lr={lr_initial}, epochs={epochs_num}")
                
                weights, coords = som_train(
                    X,
                    grid_shape=grid,
                    epochs=epochs_num,
                    sigma=sigma_initial,
                    learning_rate=lr_initial
                )
                labels = som_assign(X, weights)

                # --- Análise e Plotagem ---
                n_clusters = len(set(labels))
                score = None

                if n_clusters > 1:
                    try:
                        score = silhouette_score(X, labels)
                        print(f"  -> Silhouette Score: {score:.3f}")
                    except Exception as e:
                        print(f"  -> Erro no Silhouette: {e}")
                        score = 'Error'
                else:
                    print("  -> Silhouette não aplicável (1 cluster)")
                    score = 'N/A'
                
                results_list.append({
                    'grid_shape': f"{grid[0]}x{grid[1]}",
                    'sigma': sigma_initial,
                    'learning_rate': lr_initial,
                    'epochs': epochs_num,
                    'n_clusters': n_clusters,
                    'silhouette_score': score
                })

                # Plot
                plt.figure(figsize=(8, 6))
                for i in sorted(list(set(labels))):
                    plt.scatter(X[labels == i, 0], X[labels == i, 1], label=f"Cluster {i}")
                
                title_text = f"SOM {grid[0]}x{grid[1]} | σ={sigma_initial}, LR={lr_initial}, Epochs={epochs_num}"
                if isinstance(score, float):
                    title_text += f"\nSilhouette: {score:.3f}"

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

print("\nResultados do SOM salvos em som_results/results.csv")
