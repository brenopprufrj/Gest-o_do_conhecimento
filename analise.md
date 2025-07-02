### Análise dos Resultados de Algoritmos de Clusterização

**K-means:**

*   **Parâmetro:** `k` (número de clusters) é definido a priori.
*   **Resultados:** O score de silhueta, uma medida de quão bem definidos são os clusters, foi maior para `k=2` (0.464) e `k=3` (0.445), diminuindo à medida que `k` aumentava. Isso sugere que a estrutura natural dos dados do Iris (considerando apenas as duas primeiras features) se ajusta melhor a 2 ou 3 clusters. Visualmente, os clusters ficam menos distintos com `k` maior, com mais sobreposição.
*   **Critério de Parada:** O algoritmo para quando a mudança nos centróides é mínima, indicando que os clusters estão estáveis.

**DBSCAN:**

*   **Parâmetros:** `eps` (raio) e `minPts` (pontos mínimos) são definidos a priori.
*   **Resultados:** A principal diferença do DBSCAN é que o **número de clusters é um resultado do algoritmo, não um parâmetro de entrada**.
    *   Com `eps` pequeno (0.2), o algoritmo tende a formar mais clusters pequenos ou a classificar muitos pontos como ruído.
    *   Com `eps=0.3`, os scores de silhueta foram os melhores (em torno de 0.338), indicando uma boa separação.
    *   Com `eps` grande (0.4, 0.5), o algoritmo agrupou todos os pontos em um único cluster, tornando o score de silhueta "não aplicável". Isso acontece porque o raio de vizinhança se torna tão grande que todos os pontos se conectam.
*   **Natureza Emergente dos Clusters:** O DBSCAN encontra clusters baseados na densidade dos dados. Onde os pontos estão próximos o suficiente (definido por `eps` e `minPts`), um cluster "emerge". Isso é poderoso para encontrar clusters de formas arbitrárias e para identificar ruído, algo que o K-means não faz. A desvantagem é a sensibilidade aos parâmetros `eps` e `minPts`.

**Kohonen (SOM):**

*   **Parâmetros:** A grade, `sigma`, `learning_rate` e `epochs` são definidos a priori. O número de clusters é indiretamente influenciado pelo tamanho da grade.
*   **Resultados:** O SOM, como o K-means, força os dados a se encaixarem em um número pré-definido de neurônios (que atuam como clusters). A variação dos múltiplos parâmetros (`sigma`, `lr`, `epochs`) mostrou que o SOM pode ser bastante sensível.
    *   Grades menores (2x2) geralmente produziram scores de silhueta mais altos (em torno de 0.416), similar ao K-means com `k` baixo.
    *   Aumentar a grade (3x3, 4x4) nem sempre melhorou o resultado, às vezes levando a uma clusterização menos clara e scores mais baixos.
    *   Os outros parâmetros (`sigma`, `lr`, `epochs`) também influenciam a convergência e a qualidade final dos clusters, mas a mudança no tamanho da grade parece ter o impacto mais significativo nos scores de silhueta para este conjunto de dados.
*   **Critério de Parada:** O treinamento para após um número fixo de `epochs`.

### Conclusão e Discussão

A principal diferença observada é como o número de grupos é determinado:

1.  **K-means e SOM:** Exigem que o usuário **defina o número de clusters** (diretamente em `k` ou indiretamente pela grade do SOM). Eles sempre atribuirão cada ponto a um cluster. São bons quando se tem uma hipótese sobre quantos grupos existem nos dados.
2.  **DBSCAN:** **Descobre o número de clusters** com base na densidade dos dados. É o único dos três que pode identificar pontos como *ruído* (outliers). Isso o torna mais flexível e realista para dados do mundo real, onde o número de clusters geralmente não é conhecido e a presença de ruído é comum. A execução com `eps=0.4` e `eps=0.5` ilustra perfeitamente isso: em vez de forçar uma divisão, o DBSCAN corretamente identificou que, com esses parâmetros, todos os pontos formam uma única grande área de alta densidade.
