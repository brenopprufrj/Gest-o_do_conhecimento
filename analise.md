### Análise dos Resultados de Algoritmos de Clusterização

Este documento analisa e compara os resultados de três algoritmos de clusterização — K-means, DBSCAN e SOM — aplicados ao dataset Iris.

### Metodologia e Métricas de Avaliação

Para todos os experimentos, os algoritmos foram executados utilizando as **quatro features** do dataset Iris para calcular os clusters e as métricas de avaliação. No entanto, os gráficos gerados mostram a clusterização em apenas **duas dimensões** (comprimento e largura da sépala) para facilitar a visualização.

Foram utilizadas três métricas para avaliar a qualidade dos clusters:

1.  **Score de Silhueta (Silhouette Score):** Mede quão semelhante um objeto é ao seu próprio cluster em comparação com outros clusters. Varia de -1 a 1, onde valores mais altos indicam clusters mais densos e bem definidos.
2.  **Score de Davies-Bouldin (Davies-Bouldin Score):** Mede a similaridade média entre cada cluster e seu cluster mais semelhante. Valores mais baixos indicam uma melhor separação, com 0 sendo a pontuação ideal.
3.  **Score de Calinski-Harabasz (Calinski-Harabasz Score):** Também conhecido como critério da Razão de Variância, mede a razão entre a dispersão inter-cluster e a dispersão intra-cluster. Valores mais altos indicam clusters mais densos e bem separados.

---

### Análise por Algoritmo

**K-means:**

*   **Comportamento:** O K-means força a divisão dos dados em um número `k` de clusters pré-definido.
*   **Resultados:** O **Score de Silhueta** atingiu seu máximo em `k=2` (0.681), refletindo a clara separação entre a espécie *Setosa* e as outras duas. O score para `k=3` (0.519) também foi alto, alinhado com as três espécies do dataset. Em contraste, o **Score de Calinski-Harabasz** foi máximo em `k=4` (529.19), sugerindo que esta métrica favoreceu uma divisão em mais grupos, possivelmente mais densos. O **Score de Davies-Bouldin** foi mínimo (melhor) em `k=2` (0.404), corroborando o resultado da Silhueta.
*   **Critério de Parada:** O algoritmo para quando a mudança nos centróides é mínima.

**DBSCAN:**

*   **Comportamento:** A principal característica do DBSCAN é que o **número de clusters é um resultado do algoritmo**, não um parâmetro. Ele agrupa por densidade e é capaz de identificar ruído.
*   **Resultados:** A performance foi altamente sensível aos parâmetros `eps` e `minPts`.
    *   Com `eps` pequeno (0.2), o algoritmo tendeu a criar muitos clusters pequenos ou a classificar pontos como ruído, resultando em scores de silhueta baixos ou negativos.
    *   O melhor resultado foi com `eps=0.5` e `min_pts=5`, que encontrou **2 clusters** com um **Score de Silhueta de 0.486**, um resultado comparável ao K-means com `k=2`.
    *   Com `eps` grande (e.g., 0.4 ou 0.5 com `min_pts` baixo), o algoritmo agrupou todos os pontos em um único cluster, o que demonstra sua capacidade de não forçar uma divisão inexistente nos dados sob esses parâmetros.

**Kohonen (SOM):**

*   **Comportamento:** O SOM, assim como o K-means, força os dados a se encaixarem em um número pré-definido de neurônios (clusters), mas com a vantagem de preservar a topologia dos dados.
*   **Resultados:** O SOM mostrou-se bastante sensível à sua vasta gama de hiperparâmetros.
    *   Os melhores scores foram obtidos com uma grade **2x2** (4 neurônios), alcançando um **Score de Silhueta de 0.498**. Este resultado é muito semelhante ao do K-means com `k=4`.
    *   Aumentar a grade para 3x3 ou 4x4 (9 ou 16 neurônios) geralmente levou a uma queda nos scores, indicando um "overfitting" dos dados a mais clusters do que sua estrutura natural suporta.
*   **Critério de Parada:** O treinamento para após um número fixo de `epochs`.

---

### Resumo Comparativo

A tabela abaixo resume os melhores resultados encontrados para cada algoritmo, com base no Score de Silhueta.

| Algoritmo | Melhores Parâmetros | Nº Clusters | Score Silhueta (Maior é Melhor) | Score Davies-Bouldin (Menor é Melhor) | Score Calinski-Harabasz (Maior é Melhor) |
| :-------- | :------------------ | :---------- | :------------------------------ | :------------------------------------ | :-------------------------------------- |
| **K-means** | `k=2` | 2 | **0.681** | **0.404** | 513.92 |
| **DBSCAN** | `eps=0.5`, `min_pts=5` | 2 | 0.486 | 7.222 | 220.29 |
| **SOM** | `grid=2x2`, `sigma=1.0`, `lr=0.1`, `epochs=100` | 4 | 0.498 | 0.780 | **530.76** |

---

### Conclusão e Recomendações

A análise revela a diferença fundamental entre os algoritmos e guia a escolha da ferramenta certa para cada problema.

1.  **K-means** é ideal quando:
    *   O **número de clusters é conhecido** ou pode ser estimado.
    *   Os clusters esperados são de formato esférico e de tamanho similar.
    *   A simplicidade e velocidade de computação são importantes.

2.  **DBSCAN** é a melhor escolha quando:
    *   O **número de clusters é desconhecido**.
    *   Os clusters podem ter **formatos arbitrários** e tamanhos variados.
    *   Espera-se a presença de **ruído (outliers)** que devem ser identificados.

3.  **SOM** é particularmente útil para:
    *   **Visualização de dados de alta dimensão** em um mapa de baixa dimensão (2D).
    *   Quando a **preservação da topologia** (manter as relações de vizinhança dos dados originais) é importante.
    *   Atua de forma similar ao K-means, mas com um componente de visualização poderoso.

Em resumo, para o dataset Iris, o **K-means com k=2** produziu a clusterização com a melhor pontuação de separação, enquanto o **DBSCAN** demonstrou flexibilidade ao encontrar um número de clusters emergente e o **SOM** se destacou ao mapear os dados em uma grade, com seus melhores resultados sendo numericamente similares aos do K-means com k=4.
---

### Análise de Implementação e Escolhas de Projeto

Esta seção detalha as escolhas técnicas e a arquitetura dos scripts desenvolvidos para a experimentação.

**Estrutura Comum dos Scripts:**

Todos os scripts (`kmeans.py`, `dbscan.py`, `som.py`) seguem uma estrutura padronizada para garantir consistência e reprodutibilidade:

1.  **Bibliotecas Principais:**
    *   **NumPy:** Utilizada para todas as operações numéricas e de álgebra linear, fundamentais para os cálculos dos algoritmos.
    *   **Matplotlib:** Usada para a geração de gráficos de dispersão (scatter plots), permitindo a visualização dos clusters.
    *   **Pandas:** Empregada para estruturar os resultados dos experimentos em um DataFrame e exportá-los para um arquivo `.csv`.
    *   **Scikit-learn:** Utilizada exclusivamente para carregar o dataset Iris (`load_iris`) e para calcular as métricas de avaliação de cluster (`silhouette_score`, `davies_bouldin_score`, `calinski_harabasz_score`).
    *   **OS e Logging:** Para manipulação de diretórios e para fornecer feedback sobre o progresso da execução.

2.  **Fluxo de Execução:**
    *   **Carregamento de Dados:** O dataset Iris é carregado uma única vez. As 4 features são usadas para o cálculo (`X_full`), e as 2 primeiras (comprimento e largura da sépala) são usadas para a plotagem (`X_plot`).
    *   **Loop de Hiperparâmetros:** Cada script define um conjunto de hiperparâmetros para testar e itera sobre todas as combinações possíveis.
    *   **Execução do Algoritmo:** Em cada iteração, o algoritmo de clusterização correspondente é executado com os hiperparâmetros daquela iteração.
    *   **Cálculo de Métricas:** As métricas de avaliação são calculadas para os resultados. O código trata casos onde as métricas não são aplicáveis (e.g., quando apenas um cluster é formado).
    *   **Geração de Saída:** Para cada execução, um gráfico `.png` é salvo em um diretório específico (`/kmeans_results`, etc.), e os resultados numéricos são acumulados.
    *   **Relatório Final:** Ao final de todas as iterações, um arquivo `results.csv` é salvo no mesmo diretório, contendo um resumo completo de todos os experimentos.

---

#### K-means

*   **Implementação:** O algoritmo foi implementado **do zero (from-scratch)**. As funções `initialize_centroids`, `assign_clusters` e `update_centroids` replicam o comportamento padrão do K-means, que consiste em inicializar centróides, atribuir pontos ao centróide mais próximo e recalcular a posição do centróide com base na média dos pontos atribuídos.
*   **Hiperparâmetros Testados:** O único hiperparâmetro variado foi `k` (o número de clusters), testado para valores no intervalo `[2, 10]`.
*   **Critério de Parada:** O loop de otimização do algoritmo para quando a mudança na posição dos centróides entre iterações é menor que uma tolerância (`tol=1e-4`) ou após um número máximo de iterações.

---

#### DBSCAN

*   **Implementação:** Assim como o K-means, o DBSCAN foi implementado **do zero**. A lógica central reside nas funções `region_query`, que encontra todos os pontos vizinhos dentro de um raio `eps`, e `expand_cluster`, que expande um cluster a partir de um ponto central (core point). A implementação classifica os pontos como *core*, *border* ou *noise*.
*   **Hiperparâmetros Testados:** A experimentação foi feita em uma grade de valores para `eps` (de 0.1 a 0.7) e `min_pts` (de 2 a 10).
*   **Característica Notável:** A implementação lida corretamente com a identificação de ruído (pontos com label `-1`) e com casos onde o número de clusters resultante é 1 ou 0, o que é um comportamento esperado do DBSCAN.

---

#### SOM (Self-Organizing Map)

*   **Implementação:** O SOM também foi implementado **do zero**. A função `som_train` contém a lógica de treinamento, que inclui:
    1.  Inicialização aleatória dos pesos dos neurônios.
    2.  Um loop de treinamento por um número definido de `epochs`.
    3.  Em cada `epoch`, o algoritmo itera sobre os dados, encontra o neurônio vencedor (BMU - Best Matching Unit) e atualiza seus pesos e os de seus vizinhos.
    4.  A **taxa de aprendizado (`learning_rate`)** e o **raio de vizinhança (`sigma`)** decaem linearmente ao longo das épocas, permitindo uma convergência mais fina.
*   **Hiperparâmetros Testados:** Foi explorado o mais vasto espaço de hiperparâmetros dos três algoritmos, incluindo o formato da grade de neurônios (`grid_shape`), `sigma` inicial, `learning_rate` inicial e o número de `epochs`.
*   **Atribuição de Clusters:** Após o treinamento, a função `som_assign` atribui cada ponto do dataset ao neurônio (cluster) cujos pesos são mais próximos.