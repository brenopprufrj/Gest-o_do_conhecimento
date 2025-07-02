# Análise Comparativa de Algoritmos de Clusterização

Este repositório contém a implementação e análise de três algoritmos de clusterização clássicos: K-means, DBSCAN e SOM (Self-Organizing Maps). Os algoritmos são aplicados ao dataset Iris para comparar seu comportamento e performance sob a variação de diferentes hiperparâmetros.

## 📝 Descrição do Projeto

O objetivo deste projeto é explorar como diferentes algoritmos de clusterização particionam os dados e como seus resultados são influenciados por seus respectivos hiperparâmetros. Cada script Python corresponde a um algoritmo e foi configurado para:
1.  Executar o algoritmo com uma gama de hiperparâmetros.
2.  Calcular o **Score de Silhueta** para avaliar a qualidade da clusterização.
3.  Gerar visualizações (gráficos de dispersão) para cada combinação de parâmetros.
4.  Salvar os resultados numéricos em um arquivo CSV para análise posterior.

O dataset utilizado é o **Iris**, focando nas duas primeiras features (comprimento e largura da sépala) para facilitar a visualização em 2D.

## 🤖 Algoritmos e Parâmetros

Os seguintes algoritmos foram implementados e testados:

### 1. K-means (`kmeans.py`)
Agrupa os dados em um número *k* de clusters pré-definido.
- **Parâmetro variado:**
  - `k` (número de clusters): `[2, 3, 4, 5, 6]`

### 2. DBSCAN (`dbscan.py`)
Agrupa os dados com base na densidade, identificando clusters de formas arbitrárias e outliers. O número de clusters é um resultado emergente.
- **Parâmetros variados:**
  - `eps` (raio de vizinhança): `[0.2, 0.3, 0.4, 0.5]`
  - `minPts` (número mínimo de pontos por cluster): `[3, 4, 5]`

### 3. Kohonen (SOM) (`som.py`)
Um tipo de rede neural artificial que mapeia dados de alta dimensão para uma grade de neurônios de baixa dimensão.
- **Parâmetros variados:**
  - `Tamanho da grade`: `[(2,2), (3,3), (4,4)]`
  - `Raio de vizinhança (sigma)`: `[0.5, 1.0, 1.5]`
  - `Taxa de aprendizado (learning_rate)`: `[0.1, 0.5, 0.9]`
  - `Número de épocas (epochs)`: `[50, 100, 200]`

## 🚀 Como Executar

Para replicar os resultados, siga os passos abaixo:

1.  **Clone o repositório:**
    ```bash
    git clone <URL_DO_SEU_REPOSITORIO>
    cd <NOME_DO_REPOSITORIO>
    ```

2.  **Crie e ative um ambiente virtual:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
    *(No Windows, use `.venv\Scripts\activate`)*

3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Nota: Um arquivo `requirements.txt` pode ser criado com `pip freeze > requirements.txt` após instalar as bibliotecas abaixo).*
    Se não houver `requirements.txt`, instale manualmente:
    ```bash
    pip install numpy matplotlib scikit-learn pandas
    ```

4.  **Execute os scripts:**
    ```bash
    python3 kmeans.py
    python3 dbscan.py
    python3 som.py
    ```

## 📊 Resultados

A execução dos scripts gera os seguintes artefatos:

-   **Imagens (`.png`):** Dentro das pastas `kmeans_results/`, `dbscan_results/`, e `som_results/`, você encontrará os gráficos de clusterização para cada combinação de parâmetros.
-   **Dados (`.csv`):** Cada uma das pastas de resultados também conterá um arquivo `results.csv` com os scores de silhueta e os parâmetros utilizados, permitindo uma análise quantitativa.
-   **Análise Detalhada:** O arquivo [`analise.md`](./analise.md) contém uma discussão aprofundada sobre os resultados, comparando o comportamento dos algoritmos e destacando a diferença fundamental entre a clusterização por partição (K-means, SOM) e a baseada em densidade (DBSCAN).
