import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import streamlit as st

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

    
