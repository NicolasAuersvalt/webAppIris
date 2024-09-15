<h1>~ WebAppIris ~</h1>

## 1. INTRODUÇÃO

## 1.1 Iris Dataset

O Iris dataset, ou conjunto de dados Iris, é um dos conjuntos de dados mais conhecidos e amplamente utilizados no campo de aprendizado de máquina e estatística. Ele foi introduzido pelo biólogo e estatístico britânico Ronald A. Fisher em seu artigo intitulado “The use of multiple measurements in taxonomic problems as an example of linear discriminant analysis” em 1936 e é frequentemente usado para testes e demonstrações de técnicas de análise de dados.
Esses dados contém informações sobre 150 flores de íris, divididas em três espécies diferentes: Iris setosa, Iris versicolor e Iris virginica. Cada flor tem quatro características medidas em centímetros: comprimento da sépala, largura da sépala, comprimento da pétala e largura da pétala.

<img src="https://github.com/NicolasAuersvalt/webAppIris/blob/db288a575afb6e404ff4388e71848287c770f5b6/Imagens/1_iris-machinelearning.png">

Dessa forma, o conjunto de dados é ideal para ilustrar como diferentes técnicas podem ser usadas para separar e classificar classes distintas com base em suas características, assim como testar algoritmos, como K-Nearest Neighbors (KNN), Máquinas de Vetores de Suporte (SVM), e Redes Neurais.

## 1.2 Bibliotecas Utilizadas
Bibliotecas e Versões

 <pre><code>pip 24.0
Python 3.10.14
NumPy 2.0.1
Pandas 2.2.2
Matplotlib 3.9.2
Seaborn 0.13.2
SciPy 1.14.0
Statsmodels 0.14.2
scikit-learn 1.5.1
</code></pre>
    
## 1.3 Header

A fim de tornar os segmentos de código limpos, foi optado criar um arquivo header com a importação das bibliotecas utilizadas em cada seção. Nomeado como header.py:

<pre><code>import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.datasets import make_blobs
from scipy.stats import norm
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist
from sklearn.model_selection import KFold
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
</code></pre>


## 1.4 ChatGPT

Foi-se utilizado, sobretudo, para correlacionar as equivalências de código de R em Python, complementação de códigos como para visualização de dados, explicação de conceitos e ajustes em métodos.


## 2. ANÁLISE DESCRITIVA

## 2.1 Input

Foi importado o conjunto de dados Iris e armazenados em uma estrutura bidimensional do tipo Data Frame, como um array de duas dimensões. Em seguida, aplicada as funções “describe” e “mode”.

<pre><code>from header import *

iris_df = sns.load_dataset('iris')

# Exibe informações de média, desvio padrão, mediana (50%) 
print(iris_df.describe())
print(iris_df.mode())

</code></pre>

## 2.2 Output

## 2.2.1 Describe

O método describe() traz estatísticas descritivas que incluem aquelas que resumem a tendência central, dispersão e forma da distribuição de um conjunto de dados.

<pre><code>        sepal_length  sepal_width  petal_length  petal_width
count    150.000000   150.000000    150.000000   150.000000
mean       5.843333     3.057333      3.758000     1.199333
std        0.828066     0.435866      1.765298     0.762238
min        4.300000     2.000000      1.000000     0.100000
25%        5.100000     2.800000      1.600000     0.300000
50%        5.800000     3.000000      4.350000     1.300000
75%        6.400000     3.300000      5.100000     1.800000
max        7.900000     4.400000      6.900000     2.500000
</code></pre>

As distribuições de comprimento e largura da sépala e da pétala são relativamente normais, mas o comprimento e a largura da pétala têm variações maiores comparados à sépala. A maior variação ocorre no comprimento da pétala e largura da pétala, o que sugere que essas características podem ser mais úteis para diferenciar as espécies de flores.

## 2.2.2 Mode
O método “mode” nos retorna a moda, ou seja, o valor mais frequente em cada coluna.

<pre><code>sepal_length  sepal_width  petal_length  petal_width	species
0           5.0          3.0           1.4          0.2    	setosa
1           NaN          NaN           1.5          NaN 	versicolor
2           NaN          NaN           NaN          NaN  	virginica
</code></pre>

## 2.3 Histograma
O histograma é uma ótima maneira de visualizar a distribuição de uma variável contínua. Neste caso, ele mostra como as larguras das pétalas estão distribuídas no conjunto de dados Iris, com número de bins (intervalos) como 20. 

<pre><code>from header import *

iris_df = sns.load_dataset('iris')

# Criação do Histograma

# Tamanho da imagem
plt.figure(figsize=(8, 6))
sns.histplot(iris_df['petal_width'], kde=False, bins=20, color='skyblue')

plt.title('Histograma da Largura das Pétalas')
plt.xlabel('Largura da Pétala (cm)')
plt.ylabel('Frequência')

# Mostrar o gráfico
plt.show()
</code></pre>

<img src="https://github.com/NicolasAuersvalt/webAppIris/blob/db288a575afb6e404ff4388e71848287c770f5b6/Imagens/2_histograma_1.png">
    
A visualização mostra dois picos principais (bimodal), correspondendo às diferentes espécies do conjunto de dados. 	Espécies como Iris setosa, que tem pétalas menores, têm uma distribuição de largura de pétalas mais concentrada em valores menores, enquanto outras espécies como a virginica têm valores maiores.

## 2.4 BoxPlot
A partir dos boxplots que esse código gera, pode-se verificar as diferenças nos comprimentos e larguras das pétalas e sépalas entre as espécies, identificar possíveis outliers para cada espécie e observar a variação dos atributos dentro de cada espécie, que é representada pelo tamanho das caixas nos boxplots.

<pre><code>from header import *

iris_df = sns.load_dataset('iris')

# Criar um BoxPlot usando seaborn
plt.figure(figsize=(8, 6))
sns.boxplot(iris_df)
plt.title('BoxPlot com Seaborn')
plt.show()

</code></pre>

<img src="https://github.com/NicolasAuersvalt/webAppIris/blob/db288a575afb6e404ff4388e71848287c770f5b6/Imagens/3_boxplot_1.png">

As medidas de comprimento e largura são mais eficazes na discriminação entre as espécies, especialmente para distinguir setosa das outras duas espécies. Versicolor e virginica também podem ser diferenciadas com base nessas características, mas com menor clareza em comparação com setosa. As medidas das sépalas oferecem alguma capacidade de discriminação, mas são menos eficazes que as características das pétalas.

<pre><code>from header import *

# Carrega o dataset Iris
iris = load_iris()
iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Adicionar a coluna 'species' ao DataFrame
iris_df['species'] = iris.target

# Mapear os valores numéricos das espécies para os nomes das espécies
iris_df['species'] = iris_df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

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
plt.show()
</code></pre>

<img src="https://github.com/NicolasAuersvalt/webAppIris/blob/db288a575afb6e404ff4388e71848287c770f5b6/Imagens/4_boxplot_2.png">

Analisando-os, percebe-se que a espécie Setosa tem as características menores e menos distribuídas com alguns outliers. A Versicolor tem as características médias e a Virginica tem as características mais importantes.
Vemos que como há interseção entre as alturas de cada box, os dados não estão bem distribuídos, isto é, com separações claras, tendo dados de um cluster em outro.

## 3. AJUSTE DE DISTRIBUIÇÃO

Deve-se fazer uma padronização dos dados utilizando uma distribuição normal a fim de evitar casos em que a variável escolhida diverge muito da média (possível Overfitting ou Underfitting), os quais impedem de haver uma confiabilidade de qualquer análise tomando-os como amostra.

<pre><code># Carrega o dataset Iris
iris = load_iris()

# Selecionar a coluna da largura das sépalas (Sepal Width)

# [:,0] - Comprimento Sépala [:,1] - Largura Sépala 
# [:,2] - Comprimento Pétala | [:,3] - Largura Pétala

flor = iris.data[:, 1]

# Criar o histograma normalizado (freq = False em R equivale a density=True em Python)
plt.figure(figsize=(7, 5))
sns.histplot(flor, kde=False, bins=12 , color='skyblue', stat="density")

# Calcular a média e o desvio padrão da largura das sépalas
mean = np.mean(flor)
stnd = np.std(flor)

# Adicionar a curva de densidade normal ajustada
x_values = np.linspace(min(flor), max(flor), 100)
plt.plot(x_values, norm.pdf(x_values, mean, stnd), color='red')

plt.title('Ajuste Normal à Largura das Sépalas')
plt.xlabel('Largura das Sépalas (cm)')
plt.ylabel('Densidade')
plt.show()
 </code></pre>

Note que a coluna “[:,1]” é  Largura das Sépalas, então retira-se a média aritmética () e o desvio padrão () da mesma para o cálculo da densidade de probabilidade:


<body>
  <p>
    \( f_X(x) = \frac{1}{\sigma \sqrt{2 \pi}} e^{\frac{-(x - \mu)^2}{2 \sigma^2}} \)
  </p>
</body>
</html>


<pre><code># Load the iris dataset
iris = load_iris(as_frame=True)
df = iris.frame

# Plot settings
sns.set(style="whitegrid")

# Histograms with normal distribution curves
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

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
plt.tight_layout()
plt.show()
</code></pre>

<img src="https://github.com/NicolasAuersvalt/webAppIris/blob/db288a575afb6e404ff4388e71848287c770f5b6/Imagens/5_ajuste.png">

A partir desse gráfico, percebe-se que:
    <ul>
        <li>A maior frequência do comprimento da sépala está entre 30 e 35, que está entre 5,5 e 6.</li>
        <li>A maior frequência da largura da sépala é cerca de 70, que está entre 3,0 e 3,5.</li>
        <li>A maior frequência do comprimento da pétala é em torno de 50, que está entre 1 e 2.</li>
        <li>A maior frequência da largura da pétala está entre 40 e 50, que está entre 0,0 e 0,5.</li>
    </ul>

Além disso, conclui-se que o gráfico com melhor distribuição gaussiana é com relação à largura da sépala.
É possível concluir, portanto, onde pode-se encontrar a maior concentração de sépalas ou pétalas com determinadas características utilizando como base tal ajuste. Além disso, há alguns vales entre as frequências, indicando que possivelmente não há dados nesta região. Para validar essa hipótese, aumenta-se a quantidade de retângulos mostrados no gráfico:

<img src="https://github.com/NicolasAuersvalt/webAppIris/blob/db288a575afb6e404ff4388e71848287c770f5b6/Imagens/6_ajustenormal.png">


## 4. TÉCNICAS DE CLUSTERING

## 4.1 K-Means
Esse algoritmo é um exemplo de aplicação de machine learning supervisionado não determinístico, ou seja, é possível classificar em função do tamanho da pétala e da sépala pois sabe-se sobre o conjunto de dados em tratamento (flor, diâmetro, agrupar pétalas).
Inicialmente, serão sorteados pontos aleatórios, os quais servirão de ponto inicial para o cálculo da distância aritmética até cada ponto analisado, em seguida, substitui e ajusta o ponto utilizado para cada ponto em sequência e, por fim, retorna uma média.
O objetivo principal do algoritmo K-Means é segregar dados nos chamados Clusters, cujos dados podem ter interseções entre clusters diferentes, precisando assim ser certificada tal acurácia utilizando um método Cross Validation.


<pre><code># Configurar a semente aleatória
np.random.seed(123)

iris = load_iris()
data = iris.data  # Usando todas as quatro coluna ou [:,0], [:,1]

# Executar o algoritmo K-means com 3 clusters
kmeans = KMeans(n_clusters=3, random_state=123)
kmeans_result = kmeans.fit(data)

# Resultados
print("Centróides:\n", kmeans.cluster_centers_)
print("Rótulos:\n", kmeans.labels_)

# Visualização dos clusters
plt.figure(figsize=(6, 4))

# Scatter plot
# [:,0] - Comprimento Sépala [:,1] - Largura Sépala 
# [:,2] - Comprimento Pétala | [:,3] - Largura Pétala

plt.scatter(data[:, 0], data[:, 1], c=kmeans.labels_, cmap='viridis', marker='o')

plt.title('Clustering K-means do Dataset Iris')
plt.xlabel('Comprimento da Sépala (cm)')
plt.ylabel('Largura da Sépala (cm)') 
plt.show()
</code></pre>

<img src="https://github.com/NicolasAuersvalt/webAppIris/blob/db288a575afb6e404ff4388e71848287c770f5b6/Imagens/7_clustering.png">

A partir do gráfico acima, infere-se que a espécie Setosa tem pétalas com comprimentos e larguras menores, Versicolor está no meio das outras duas espécies em termos de comprimento e largura da pétala e Virginica tem o maior comprimento e largura de pétalas.

<img src="https://github.com/NicolasAuersvalt/webAppIris/blob/db288a575afb6e404ff4388e71848287c770f5b6/Imagens/8_clustering.png">

A partir do gráfico acima, percebe-se que a espécie Setosa tem tamanhos de sépalas menores, mas larguras de sépalas maiores. A Versicolor está no meio das outras duas espécies em termos de comprimento e largura da sépala. E a espécie Virginica tem tamanhos de sépala maiores, mas larguras de sépala menores.

## 4.2 Hierarchical Clustering (HC Clustering)
O agrupamento hierárquico é outro método de aprendizagem não supervisionado para agrupar pontos de dados.
Inicialmente, o método procura agrupar pontos próximos sucessivamente, até que haja N clusters restante. Então, o algoritmo pode calcular os métodos de ligação das seguintes formas:
1. A menor distância entre os pontos mais próximos

<img src="https://github.com/NicolasAuersvalt/webAppIris/blob/db288a575afb6e404ff4388e71848287c770f5b6/Imagens/9_distancia.png">

2. A distância entre os pontos mais distantes

<img src="https://github.com/NicolasAuersvalt/webAppIris/blob/db288a575afb6e404ff4388e71848287c770f5b6/Imagens/10_distancia.png">

3. A distância média entre os clusters é calculada pela soma entre cada par e depois dividida pelo número total de conjuntos de dados.

4. Distância entre os centróides

<img src="https://github.com/NicolasAuersvalt/webAppIris/blob/db288a575afb6e404ff4388e71848287c770f5b6/Imagens/11_distancia.png">

É possível visualizar as distâncias através de um Dendrograma, que exemplifica, similarmente a uma árvore, a distância entre os pontos mais próximos (mostrado na parte inferior) e cresce conforme diminui os agrupamentos (não mais pontos isolados).

<img src="https://github.com/NicolasAuersvalt/webAppIris/blob/db288a575afb6e404ff4388e71848287c770f5b6/Imagens/12_dendograma.png">

<pre><code># Carregar o dataset Iris
iris = load_iris()
data = iris.data  # Usando todas as quatro features (colunas) como em R

# Calcular a matriz de distância
distance_matrix = pdist(data, metric='euclidean')

# Executar o algoritmo de clustering hierárquico
hc_result = linkage(distance_matrix, method='ward')

# Plotar o dendrograma
plt.figure(figsize=(10, 7))
dendrogram(hc_result)

# Adicionar título e rótulos
plt.title('Dendrograma do Clustering Hierárquico - Dataset Iris')
plt.xlabel('Índice das Amostras')
plt.ylabel('Distância')
plt.show()
</code></pre>

<img src="https://github.com/NicolasAuersvalt/webAppIris/blob/db288a575afb6e404ff4388e71848287c770f5b6/Imagens/13_HC.png">

<pre><code># Carregar o dataset Iris
iris = load_iris()
data = iris.data  # Usando todas as quatro features (colunas)
feature_names = iris.feature_names

# Usar apenas as duas primeiras colunas para o gráfico
data1 = data[:, 2]  # Comprimento do sépala (Sepal Length)
data2 = data[:, 3]  # Largura do sépala (Sepal Width)

# Calcular a matriz de distância
distance_matrix = pdist(data, metric='euclidean')

# Executar o algoritmo de clustering hierárquico
hc_result = linkage(distance_matrix, method='ward')

# Determinar o número de clusters
num_clusters = 3  # O dataset Iris tem 3 espécies
clusters = fcluster(hc_result, num_clusters, criterion='maxclust')

# Plotar o gráfico de dispersão
plt.figure(figsize=(10, 7))
scatter = plt.scatter(data1, data2, c=clusters, cmap='viridis', edgecolor='k', s=50)

# Adicionar título e rótulos
plt.title('Gráfico de Dispersão com Clustering Hierárquico - Dataset Iris')
plt.xlabel('Comprimento da Pétala')  # Sepal Length
plt.ylabel('Largura da Pétala') # Sepal Width

# Mostrar o gráfico
plt.show()
</code></pre>

<img src="https://github.com/NicolasAuersvalt/webAppIris/blob/db288a575afb6e404ff4388e71848287c770f5b6/Imagens/14_hc.png">

Roxo = Iris Setosa; Amarelo = Iris Versicolor; Verde = Iris Virginica.

Note que há interseção entre os clusters, indicando que o método não é absolutamente perfeito e pode ser avaliado através do Cross Validation.

## 5. CROSS-VALIDATION: K-FOLD

Para avaliar a acurácia dos métodos de clusterização, é selecionada uma porcentagem de dados para treinamento e o restante para teste (80:20).

<img src="https://github.com/NicolasAuersvalt/webAppIris/blob/db288a575afb6e404ff4388e71848287c770f5b6/Imagens/15_cross-validation.png">

## 6. K-NN (K-Nearest Neighbour)

Seleciona-se a partir de uma clusterização 80% dos pontos divididos em 4 folders, tendo 20% em cada um. A partir disso, sorteia ou seleciona algum ponto (de preferência pontos críticos ou muito distante dos clusters), calcula a distância entre ele e K pontos rotulados (clusterizados) próximos a ele. Então, rotula este ponto em função da distância e da quantidade de pontos próximos a ele, e depois compara com o rótulo do ajuste. 
Em sequência, seleciona outro folder para teste e retém os outros para treinamento, mantendo este ciclo até que tenha sido considerado os 5 folders como treino.

<img src="https://github.com/NicolasAuersvalt/webAppIris/blob/db288a575afb6e404ff4388e71848287c770f5b6/Imagens/16_knn.png">

## 7. CONFUSION MATRIX

A fim de se avaliar os dados, é possível utilizar previsões, possuindo as informações reais da classificação, do que é esperado ou não, e calcular essas previsões. 
Por exemplo, caso uma espécime tenha sido classificada como virginica, mas esperava-se setosa (pressupõe o dado real para avaliação), então há um desvio de resultado no modelo, estando aberto a correções e ajustes.

<img src="https://github.com/NicolasAuersvalt/webAppIris/blob/db288a575afb6e404ff4388e71848287c770f5b6/Imagens/17_confusion.png">

## 7.1 Input
<pre><code>from header import *

# Carregar o conjunto de dados Iris
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Dividir os dados em treinamento e teste
np.random.seed(123)  # Para reprodutibilidade
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

# Normalizar os dados
scaler = MinMaxScaler()
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized = scaler.transform(X_test)

# Treinar o modelo KNN
k = 3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train_normalized, y_train)

# Fazer previsões
y_pred = knn.predict(X_test_normalized)

# Avaliar o modelo
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print(f"Confusion Matrix:\n{conf_matrix}")
print(f"Acurácia: {accuracy:.4f}")

# Executar validação cruzada
cv_scores = cross_val_score(knn, X, y, cv=10)  # 10-fold cross-validation

# Calcular média e desvio padrão da acurácia
mean_accuracy = np.mean(cv_scores)
std_accuracy = np.std(cv_scores)

print(f"Média da Acurácia do Cross-Validation): {mean_accuracy:.4f}")
print(f"Desvio Padrão do Cross-Validation): {std_accuracy:.4f}")
 </code></pre>


## 7.2 Output
<pre><code>Confusion Matrix:
[[15  0  0]
 [ 0 13  2]
 [ 0  0 15]]
Acurácia: 0.9556
Média da Acurácia do Cross-Validation): 0.9667
Desvio Padrão do Cross-Validation): 0.0447
 </code></pre>

## 8. REFERÊNCIAS
<ul>
    <li><a href="https://akiradev.netlify.app/posts/machine-learning/" target="_blank">https://akiradev.netlify.app/posts/machine-learning/</a></li>
    <li><a href="https://www.w3schools.com/python/pandas/" target="_blank">https://www.w3schools.com/python/pandas/</a></li>
    <li>LEDO, Luiz. Manual didático em Machine Learning com aplicações em R e Python.</li>
    <li><a href="https://en.wikipedia.org/wiki/Generalization_error" target="_blank">https://en.wikipedia.org/wiki/Generalization_error</a></li>
    <li><a href="https://www.youtube.com/watch?v=vBtuL2U9xic" target="_blank">https://www.youtube.com/watch?v=vBtuL2U9xic</a></li>
    <li><a href="https://www.javatpoint.com/hierarchical-clustering-in-machine-learning" target="_blank">https://www.javatpoint.com/hierarchical-clustering-in-machine-learning</a></li>
    <li><a href="https://www.displayr.com/what-is-dendrogram/" target="_blank">https://www.displayr.com/what-is-dendrogram/</a></li>
    <li><a href="https://www.javatpoint.com/cross-validation-in-machine-learning" target="_blank">https://www.javatpoint.com/cross-validation-in-machine-learning</a></li>
    <li><a href="https://www.javatpoint.com/confusion-matrix-in-machine-learning" target="_blank">https://www.javatpoint.com/confusion-matrix-in-machine-learning</a></li>
    <li><a href="https://www.javatpoint.com/k-nearest-neighbor-algorithm-for-machine-learning" target="_blank">https://www.javatpoint.com/k-nearest-neighbor-algorithm-for-machine-learning</a></li>
    <li><a href="https://www.javatpoint.com/overfitting-and-underfitting-in-machine-learning" target="_blank">https://www.javatpoint.com/overfitting-and-underfitting-in-machine-learning</a></li>
    <li><a href="https://www.chatgpt.com/" target="_blank">https://www.chatgpt.com/</a></li>
</ul>
