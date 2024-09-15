from header import *

def box_plot():
    st.title("Histograma")

    # Carregar o dataset iris
    iris = load_iris()
    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    iris_df['species'] = iris.target
    iris_df['species'] = iris_df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

    st.write("O histograma é uma ótima maneira de visualizar a distribuição de uma variável contínua. Neste caso, ele mostra como as larguras das pétalas estão distribuídas no conjunto de dados Iris, com número de bins (intervalos) como 20.")

    # Criar o histograma da largura das pétalas
    plt.figure(figsize=(8, 6))
    sns.histplot(iris_df['petal width (cm)'], kde=False, bins=20, color='skyblue')
    plt.title('Histograma da Largura das Pétalas')
    plt.xlabel('Largura da Pétala (cm)')
    plt.ylabel('Frequência')
    st.pyplot(plt)

    st.write("A visualização mostra dois picos principais (bimodal), correspondendo às diferentes espécies do conjunto de dados. Espécies como Iris setosa, que tem pétalas menores, têm uma distribuição de largura de pétalas mais concentrada em valores menores, enquanto outras espécies como a virginica têm valores maiores.")

    st.title("BoxPlot")

    st.write("A partir dos boxplots que esse código gera, pode-se verificar as diferenças nos comprimentos e larguras das pétalas e sépalas entre as espécies, identificar possíveis outliers para cada espécie e observar a variação dos atributos dentro de cada espécie, que é representada pelo tamanho das caixas nos boxplots.")

    st.write("As medidas de comprimento e largura são mais eficazes na discriminação entre as espécies, especialmente para distinguir setosa das outras duas espécies. Versicolor e virginica também podem ser diferenciadas com base nessas características, mas com menor clareza em comparação com setosa. As medidas das sépalas oferecem alguma capacidade de discriminação, mas são menos eficazes que as características das pétalas.")

    # Definir a paleta de cores
    palette = {"setosa": "skyblue", "versicolor": "darkorange", "virginica": "tomato"}

    # Criar a figura com subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # BoxPlot para 'petal length (cm)'
    sns.boxplot(x='species', y='petal length (cm)', data=iris_df, palette=palette, ax=axes[0, 0])
    axes[0, 0].set_title('Comprimento da Pétala por Espécie')

    # BoxPlot para 'petal width (cm)'
    sns.boxplot(x='species', y='petal width (cm)', data=iris_df, palette=palette, ax=axes[0, 1])
    axes[0, 1].set_title('Largura da Pétala por Espécie')

    # BoxPlot para 'sepal length (cm)'
    sns.boxplot(x='species', y='sepal length (cm)', data=iris_df, palette=palette, ax=axes[1, 0])
    axes[1, 0].set_title('Comprimento da Sépala por Espécie')

    # BoxPlot para 'sepal width (cm)'
    sns.boxplot(x='species', y='sepal width (cm)', data=iris_df, palette=palette, ax=axes[1, 1])
    axes[1, 1].set_title('Largura da Sépala por Espécie')

    
    # Ajustar o layout para evitar sobreposição
    plt.tight_layout()

    # Mostrar a figura
    st.pyplot(plt)

    st.write("Analisando-os, percebe-se que a espécie Setosa tem as características menores e menos distribuídas com alguns outliers. A Versicolor tem as características médias e a Virginica tem as características mais importantes. Vemos que como há interseção entre as alturas de cada box, os dados não estão bem distribuídos, isto é, com separações claras, tendo dados de um cluster em outro.")

def kmeans_clustering():
    st.title('Clustering K-means do Dataset Iris')

    # Carregando o dataset Iris
    iris = load_iris()
    data = iris.data

    # Aplicando o algoritmo K-Means
    kmeans = KMeans(n_clusters=3, random_state=123)
    kmeans_result = kmeans.fit(data)

    # Comentário sobre o algoritmo
    st.write("    Esse algoritmo é um exemplo de aplicação de machine learning não supervisionado, ou seja, é possível classificar em função do tamanho da pétala e da sépala, pois sabe-se sobre o conjunto de dados em tratamento. Inicialmente, serão sorteados pontos aleatórios, que servirão de ponto inicial para o cálculo da distância aritmética até cada ponto analisado, em seguida, substitui e ajusta o ponto utilizado para cada ponto em sequência e, por fim, retorna uma média. O objetivo principal do algoritmo K-Means é segregar dados nos chamados Clusters, cujos dados podem ter interseções entre clusters diferentes, precisando assim ser certificada tal acurácia utilizando um método de Cross Validation.")

    # Exibindo os centróides
    st.write("Centróides:\n", kmeans.cluster_centers_)

    # Plotando o gráfico de clusters baseado em comprimento e largura da sépala
    plt.figure(figsize=(6, 4))
    plt.scatter(data[:, 0], data[:, 1], c=kmeans.labels_, cmap='viridis', marker='o')
    plt.title('Clustering K-means do Dataset Iris')
    plt.xlabel('Comprimento da Sépala (cm)')
    plt.ylabel('Largura da Sépala (cm)')
    st.pyplot(plt)

    # Comentário sobre o gráfico de sépala
    st.write("    A partir do gráfico acima, infere-se que a espécie Setosa tem pétalas com comprimentos e larguras menores. Versicolor está no meio das outras duas espécies em termos de comprimento e largura da pétala, e Virginica tem o maior comprimento e largura de pétalas.")

    # Plotando o gráfico de clusters baseado em comprimento e largura da pétala
    plt.figure(figsize=(6, 4))
    plt.scatter(data[:, 2], data[:, 3], c=kmeans.labels_, cmap='viridis', marker='o')
    plt.title('Clustering K-means do Dataset Iris')
    plt.xlabel('Comprimento da Pétala (cm)')
    plt.ylabel('Largura da Pétala (cm)')
    st.pyplot(plt)

    # Comentário sobre o gráfico de pétala
    st.write("    A partir do gráfico acima, infere-se que a espécie Setosa tem pétalas com comprimentos e larguras menores. Versicolor está no meio das outras duas espécies em termos de comprimento e largura da pétala, e Virginica tem o maior comprimento e largura de pétalas.")


def scatter_with_hc_clusters():
    st.title('Gráfico de Dispersão com Clustering Hierárquico')

    # Comentários
    st.write("    O agrupamento hierárquico é outro método de aprendizagem não supervisionado para agrupar pontos de dados. Inicialmente, o método procura agrupar pontos próximos sucessivamente, até que haja N clusters restante. Então, o algoritmo pode calcular os métodos de ligação das seguintes formas:")

    iris = load_iris()
    data = iris.data
    data1 = data[:, 2]
    data2 = data[:, 3]
    distance_matrix = pdist(data, metric='euclidean')
    hc_result = linkage(distance_matrix, method='ward')
    num_clusters = 3
    clusters = fcluster(hc_result, num_clusters, criterion='maxclust')

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(data1, data2, c=clusters, cmap='viridis', edgecolor='k', s=50)
    plt.title('Gráfico de Dispersão com Clustering Hierárquico - Dataset Iris')
    plt.xlabel('Comprimento da Pétala')
    plt.ylabel('Largura da Pétala')
    st.pyplot(plt)

    st.write("    É possível visualizar as distâncias através de um Dendograma, que exemplifica, similarmente a uma árvore, a distância entre os pontos mais próximos (mostrado na parte inferior) e cresce conforme diminui os agrupamentos (não mais pontos isolados).")

    iris = load_iris()
    data = iris.data
    distance_matrix = pdist(data, metric='euclidean')
    hc_result = linkage(distance_matrix, method='ward')

    # Comentários

    plt.figure(figsize=(10, 7))
    dendrogram(hc_result)
    plt.title('Dendrograma do Clustering Hierárquico - Dataset Iris')
    plt.xlabel('Índice das Amostras')
    plt.ylabel('Distância')
    st.pyplot(plt)



import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import load_iris
from scipy.stats import norm

def iris_histograms():
    st.title('Histogramas com Ajuste Normal do Dataset Iris')

    # Carregando o dataset Iris
    iris = load_iris(as_frame=True)
    df = iris.frame

    # Criando uma figura com 4 subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    st.write("Deve-se fazer uma padronização dos dados utilizando uma distribuição normal a fim de evitar casos em que a variável escolhida diverge muito da média (possível Overfitting ou Underfitting), os quais impedem de haver uma confiabilidade de qualquer análise tomando-os como amostra.")

    # Petal Width
    sns.histplot(df['petal width (cm)'], kde=False, stat='density', ax=axs[0, 0])
    mu, std = df['petal width (cm)'].mean(), df['petal width (cm)'].std()
    xmin, xmax = axs[0, 0].get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    axs[0, 0].plot(x, p, 'r', lw=2)
    axs[0, 0].set_title('Ajuste Normal - Petal Width')

    # Petal Length
    sns.histplot(df['petal length (cm)'], kde=False, stat='density', ax=axs[0, 1])
    mu, std = df['petal length (cm)'].mean(), df['petal length (cm)'].std()
    xmin, xmax = axs[0, 1].get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    axs[0, 1].plot(x, p, 'r', lw=2)
    axs[0, 1].set_title('Ajuste Normal - Petal Length')

    # Sepal Width
    sns.histplot(df['sepal width (cm)'], kde=False, stat='density', ax=axs[1, 0])
    mu, std = df['sepal width (cm)'].mean(), df['sepal width (cm)'].std()
    xmin, xmax = axs[1, 0].get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    axs[1, 0].plot(x, p, 'r', lw=2)
    axs[1, 0].set_title('Ajuste Normal - Sepal Width')

    # Sepal Length
    sns.histplot(df['sepal length (cm)'], kde=False, stat='density', ax=axs[1, 1])
    mu, std = df['sepal length (cm)'].mean(), df['sepal length (cm)'].std()
    xmin, xmax = axs[1, 1].get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = norm.pdf(x, mu, std)
    axs[1, 1].plot(x, p, 'r', lw=2)
    axs[1, 1].set_title('Ajuste Normal - Sepal Length')

    # Ajustar layout e mostrar o gráfico
    plt.tight_layout()
    st.pyplot(fig)

    # Comentário sobre o gráfico
    st.write("A partir desse gráfico, percebe-se que:\n"
             "1. A maior frequência do comprimento da sépala está entre 5,5 e 6 cm.\n"
             "2. A maior frequência da largura da sépala é entre 3,0 e 3,5 cm.\n"
             "3. A maior frequência do comprimento da pétala é entre 1 e 2 cm.\n"
             "4. A maior frequência da largura da pétala é entre 0,0 e 0,5 cm.\n"
             "Além disso, conclui-se que o gráfico com melhor distribuição gaussiana é o da largura da sépala. É possível concluir onde pode-se encontrar a maior concentração de sépalas ou pétalas com determinadas características utilizando como base tal ajuste.")
    st.write("Além disso, há alguns vales entre as frequências, indicando que possivelmente não há dados nesta região. Para validar essa hipótese, aumenta-se a quantidade de retângulos mostrados no gráfico.")

    # Criar o histograma normalizado (freq = False em R equivale a density=True em Python)
    plt.figure(figsize=(7, 5))
    sns.histplot(df['sepal width (cm)'], kde=False, bins=30, color='skyblue', stat="density")

    # Calcular a média e o desvio padrão da largura das sépalas
    mean = np.mean(df['sepal width (cm)'])
    stnd = np.std(df['sepal width (cm)'])

    # Adicionar a curva de densidade normal ajustada
    x_values = np.linspace(df['sepal width (cm)'].min(), df['sepal width (cm)'].max(), 100)
    plt.plot(x_values, norm.pdf(x_values, mean, stnd), color='red')

    # Adicionar título e rótulos
    plt.title('Ajuste Normal à Largura das Sépalas')
    plt.xlabel('Largura das Sépalas (cm)')
    plt.ylabel('Densidade')
    st.pyplot(plt)

if __name__ == "__main__":
    iris_histograms()


def knn_classifier():
    st.title('Classificador KNN do Dataset Iris')

    st.write("Para avaliar a acurácia dos métodos de clusterização, é selecionada uma porcentagem de dados para treinamento e o restante para teste (80:20).")

    st.image(r'Imagens/15_cross-validation.png', use_column_width=False, width=600)


    st.write("Seleciona-se a partir de uma clusterização 80% dos pontos divididos em 4 folders, tendo 20% em cada um. A partir disso, sorteia ou seleciona algum ponto (de preferência pontos críticos ou muito distante dos clusters), calcula a distância entre ele e K pontos rotulados (clusterizados) próximos a ele. Então, rotula este ponto em função da distância e da quantidade de pontos próximos a ele, e depois compara com o rótulo do ajuste. Em sequência, seleciona outro folder para teste e retém os outros para treinamento, mantendo este ciclo até que tenha sido considerado os 5 folders como treino.")

    st.image(r'Imagens/16_knn.png', use_column_width=False, width=500)
    
    st.write("A fim de se avaliar os dados, é possível utilizar previsões, possuindo as informações reais da classificação, do que é esperado ou não, e calcular essas previsões. Por exemplo, caso uma espécime tenha sido classificada como virginica, mas esperava-se setosa (pressupõe o dado real para avaliação), então há um desvio de resultado no modelo, estando aberto a correções e ajustes.")
    
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
    scaler = MinMaxScaler()
    X_train_normalized = scaler.fit_transform(X_train)
    X_test_normalized = scaler.transform(X_test)
    k = 3
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_normalized, y_train)
    y_pred = knn.predict(X_test_normalized)
    conf_matrix = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    
    # Exibindo os centróides
    # st.write("Centróides:\n", kmeans.cluster_centers_)

    
    st.write("Confusion Matrix:\n")
    st.write(conf_matrix)
    
    st.write(f"Acurácia: {accuracy:.4f}")

    cv_scores = cross_val_score(knn, X, y, cv=10)
    mean_accuracy = np.mean(cv_scores)
    std_accuracy = np.std(cv_scores)

    st.write(f"Média da Acurácia do Cross-Validation: {mean_accuracy:.4f}")
    st.write(f"Desvio Padrão do Cross-Validation: {std_accuracy:.4f}")
