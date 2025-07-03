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