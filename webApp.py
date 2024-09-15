from header import *
from calculadora import *
from iris import *

def index():
    st.title("WebAppIris")

    # Texto básico
    st.write("""
    O Iris dataset, ou conjunto de dados Iris, é um dos conjuntos de dados mais conhecidos e amplamente utilizados no campo de aprendizado de máquina e estatística. Ele foi introduzido pelo biólogo e estatístico britânico Ronald A. Fisher em seu artigo intitulado “The use of multiple measurements in taxonomic problems as an example of linear discriminant analysis” em 1936 e é frequentemente usado para testes e demonstrações de técnicas de análise de dados.\n""")
    st.write("Esses dados contém informações sobre 150 flores de íris, divididas em três espécies diferentes: Iris setosa, Iris versicolor e Iris virginica. Cada flor tem quatro características medidas em centímetros: comprimento da sépala, largura da sépala, comprimento da pétala e largura da pétala.")

    # Anexar uma imagem
    st.image(r'Imagens/1_iris-machinelearning.png', use_column_width=True)


# Menu Principal
def main():
    st.sidebar.title("Menu")
    page = st.sidebar.selectbox("Escolha uma Página:", 
                                ["Página Inicial", 
                                 "Projeto Iris", 
                                 "Calculadoras"])

    if page == "Página Inicial":
        index()

    elif page == "Projeto Iris":
        iris_option = st.sidebar.selectbox("Escolha uma Opção:",
                                           ["Histograma e BoxPlot",
                                            "Clustering K-means",
                                            "Clustering Hierárquico",
                                            "Histogramas com Ajuste Normal do Dataset Iris",
                                            "Classificador KNN"])

        if iris_option == "Clustering K-means":
            kmeans_clustering()
        elif iris_option == "Histograma e BoxPlot":
            box_plot()
        elif iris_option == "Clustering Hierárquico":
            scatter_with_hc_clusters()
        elif iris_option == "Histogramas com Ajuste Normal do Dataset Iris":
            iris_histograms()
        elif iris_option == "Classificador KNN":
            knn_classifier()

    elif page == "Calculadoras":
        calc_option = st.sidebar.selectbox("Escolha uma Calculadora:", 
                                           ["Intervalo de Confiança para Média",
                                            "Intervalo de Confiança para Proporção",
                                            "Tamanho da Amostra para a Média", 
                                            "Tamanho da Amostra para Proporção",
                                            "Teste de Hipótese para Média",
                                            "Regressão Linear Simples"
                                           ])

        if calc_option == "Intervalo de Confiança para Média":
            calcular_intervalo_confianca()
        elif calc_option == "Intervalo de Confiança para Proporção":
            calculadora_proporcao()
        elif calc_option == "Tamanho da Amostra para a Média":
            calcular_tamanho_amostra_media()
        elif calc_option == "Tamanho da Amostra para Proporção":
            calcular_tamanho_amostra_proporcao()
        elif calc_option == "Teste de Hipótese para Média":
            hipotese_media()
        elif calc_option == "Regressão Linear Simples":
            regressao_linear_simples()

if __name__ == "__main__":
    main()
