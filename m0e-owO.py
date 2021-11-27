# Nomes e RAs:
# Gabriel Penajo Machado,    769712
# Matheus Ramos de Carvalho, 769703
# Matheus Teixeira Matioli,  769783

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.cluster import Ward
from sklearn import preprocessing
from collections import defaultdict
from sklearn.metrics import silhouette_samples, silhouette_score

# dataset src:
# https://archive.ics.uci.edu/ml/datasets/UrbanGB%2C+urban+road+accidents+coordinates+labelled+by+the+urban+center


def silhouette(X, y, n_clusters):
    """
    Calcula e 
    plota os scores de silhueta.

    :param X: conjunto de dados.
    :param y: agrupamento obtido após execução de um Algoritmo de Agrupamento.
    :param kmeans: objeto da classe KMeans aplicado a um dataset X
    :param n_clusters: número de clusters para execução do KMeans
    """
    fig, (ax1) = plt.subplots(1, 1)
    fig.set_size_inches(18, 7)
    ax1.set_xlim([-1, 1])
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    silhouette_avg = silhouette_score(X, y)
    print(
        "For n_clusters =",
        n_clusters,
        "The average silhouette_score is :",
        silhouette_avg,
    )
    sample_silhouette_values = silhouette_samples(X, y)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[y == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        ax1.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            ith_cluster_silhouette_values,
            alpha=0.7,
        )

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples
    
    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-1, -0.8, -0.6, -0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.show()


def preprocess(df):
    """
    Pré-processamento do DataFrame

    :param df: o DataFrame utilizado.
    :returns: a variável X (Dataset préprocessado).
    """
    # Diminuindo o tamanho do dataset por questões de escalabilidade
    X = df.sample(frac=0.1, random_state=42)
    X.reset_index(inplace=True, drop=True)

    # Conforme recomendado pelos donos do dataset, reduzimos a escala da
    # longitude em 1,7.
    X['longitude'] = X['longitude'].apply(lambda x: x/1.7)
    
    # Normalização das coordenadas.
    scaler = preprocessing.StandardScaler()
    X[col] = scaler.fit_transform(X[col])
    
    return X


def kmeans_validation(X):
    """
    Plotagem da Soma dos Quadrados do Erro para decisão do k-ideal seguindo o
    Método do "Cotovelo".

    :param X: conjunto de dados
    """
    # Algoritmo kmeans e validação
    cotovelo = []
    k_range = range(2,15,2)
    for j in k_range:
        print(f'fazendo para {j} clusters..')
        kmeans = KMeans(n_clusters=j, random_state=42).fit(X)
        cotovelo.append(kmeans.inertia_)

    # Plotagem dos resultados
    plt.plot(k_range, cotovelo, marker='x')
    plt.xlabel('Número de clusters (k)')
    plt.ylabel('Soma dos quadrados do erro')
    plt.show()


def kmeans_exec(X, cluster_n=1):
    """
    Execução do algoritmo KMeans após escolha do número de clusters.

    :param X: conjunto de dados.
    :param cluster_n: número de agrupamentos.
    :returns: y (o rótulo associado a cada instância)
    """
    kmeans = KMeans(n_clusters=cluster_n, random_state=42).fit(X)
    y = kmeans.predict(X)
    
    # Validação usando o score de silhueta
    silhouette(X, y, cluster_n)

    return y


def plot_clustering(X, y, cluster_n=1):
    """
    Plotagem do resultado do agrupamento.

    :param X: conjunto de dados.
    :param y: agrupamento obtido após execução de um Algoritmo de Agrupamento.
    """
    grupos_x = defaultdict(list)
    grupos_y = defaultdict(list)
    
    for j in range(len(X)):
        grupos_x[y[j]].append(X.loc[j]['longitude'])
        grupos_y[y[j]].append(X.loc[j]['latitude'])

    for i in range(cluster_n):
        plt.scatter(grupos_x[i], grupos_y[i], s=5) 
    plt.show()


# Função main, na qual o usuário faz as escolhas do programa.
if __name__ == '__main__':
    # Leitura do arquivo csv pelo pandas e inserção das headers "longitude" e "latitude".
    col = ["longitude","latitude"]
    df = pd.read_csv('UrbanGB.txt',header=None, names=col)
    
    # Conjunto de dados propriamente dito
    X = preprocess(df)
    # Plotagem da Soma do Quadrado do Erro
    kmeans_validation(X)

    # Número ideal de clusters após análise
    k = 6
    # Execução do KMeans
    y = kmeans_exec(X, cluster_n=k)
    # Plotagem dos resultados do KMeans para k=6
    plot_clustering(X, y, cluster_n=k)

    # clustering = DBSCAN(eps=3, min_samples=2).fit(df)
    # print(clustering.labels_)