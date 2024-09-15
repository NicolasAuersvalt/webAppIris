from header import *

# Funções para a calculadora
def calcular_intervalo_confianca():
    st.title('Calculadora de Intervalo de Confiança para Média')
    st.write("Sugestão: Suponha que você esteja conduzindo um estudo sobre o tempo médio que os estudantes levam para completar um teste. Você coleta uma amostra aleatória de 30 estudantes e registra seus tempos em minutos. A média amostral é de 25 minutos e o desvio padrão amostral é de 5 minutos. Construa um intervalo de confiança de 95% para a média populacional do tempo que os estudantes levam para completar o teste.")

    media = st.number_input("Média:", min_value=-100.0, max_value=100.0, value=1.0, format="%.6f")
    confiança = st.number_input("Valor de Confiança (%) :", min_value=0.0, max_value=100.0, value=95.0, format="%.6f")
    sd = st.number_input("Desvio Padrão:", min_value=0.1, max_value=1000.0, value=1.0, format="%.6f")
    n = st.number_input("Tamanho da amostra:", min_value=1, max_value=1000, value=30)

    # Comentários
    

    z = abs(norm.ppf((1 + (confiança / 100)) / 2))
    LI = media - (z * sd / np.sqrt(n))
    LS = media + (z * sd / np.sqrt(n))

    st.text(f"Limite inferior: {LI:.6f}")
    st.text(f"Limite superior: {LS:.6f}")

    fig, ax = plt.subplots()
    ax.barh(['Média'], [media], xerr=[(LS - LI) / 2], color='skyblue', alpha=0.7)
    ax.set_xlim(LI - (LS - LI), LS + (LS - LI))
    ax.axvline(x=LI, color='red', linestyle='--', label=f'Limite Inferior ({LI:.6f})')
    ax.axvline(x=LS, color='green', linestyle='--', label=f'Limite Superior ({LS:.6f})')
    ax.legend()
    st.pyplot(fig)

def calculadora_proporcao():
    st.title('Calculadora de Intervalo de Confiança para Proporção')
    st.write("Sugestão: Em uma pesquisa de opinião, 500 entrevistados foram questionados sobre sua preferência por um determinado produto. 300 deles expressaram uma preferência pelo produto. Construa um intervalo de confiança de 90% para a proporção populacional de pessoas que preferem o produto.")

    p = st.number_input('p :', min_value=0.0, max_value=1.0, value=0.5, format="%.6f")
    conf = st.number_input('Valor de Confiança (%) :', min_value=0.0, max_value=100.0, value=95.0, format="%.6f")
    n = st.number_input('Tamanho da amostra :', min_value=1, max_value=1000, value=30)

    # Comentários
    

    z = abs(norm.ppf((1 + (conf / 100)) / 2))

    if 0 <= p <= 1:
        LI = p - (z * np.sqrt((p * (1 - p)) / n))
        LS = p + (z * np.sqrt((p * (1 - p)) / n))

        st.write(f'Limite inferior: {LI:.6f}')
        st.write(f'Limite superior: {LS:.6f}')

        fig, ax = plt.subplots()
        ax.barh(['Proporção'], [p], xerr=[(LS - LI) / 2], color='skyblue', alpha=0.7)
        ax.set_xlim(LI - (LS - LI), LS + (LS - LI))
        ax.axvline(x=LI, color='red', linestyle='--', label=f'Limite Inferior ({LI:.6f})')
        ax.axvline(x=LS, color='green', linestyle='--', label=f'Limite Superior ({LS:.6f})')
        ax.legend()
        st.pyplot(fig)
    else:
        st.write('Erro: o valor de p deve estar entre 0 e 1.')

def calcular_tamanho_amostra_media():
    st.title('Calculadora do Tamanho da Amostra para a Média')

    st.write("Sugestão: Suponha que você deseje estimar a média de altura dos alunos de uma escola com um nível de confiança de 95% e uma margem de erro de 2 centímetros. O desvio padrão da altura populacional é conhecido como 10 centímetros. Quantos alunos você precisa incluir na amostra para obter essa estimativa?")

    margem_erro = st.number_input('Margem de Erro :', min_value=0.1, max_value=100.0, value=1.0)
    conf = st.number_input('Valor de Confiança (%) :', min_value=0.0, max_value=100.0, value=95.0)
    desvio_padrao = st.number_input('Desvio Padrão :', min_value=0.1, max_value=1000.0, value=1.0)

    # Comentários
    

    z = abs(norm.ppf((1 + (conf / 100)) / 2))
    tamanho_amostra = ((z * desvio_padrao) / margem_erro) ** 2

    st.write(f'Tamanho da Amostra para Média: {tamanho_amostra:.2f}')
    
    

def calcular_tamanho_amostra_proporcao():
    st.title('Calculadora do Tamanho da Amostra para Proporção')
    st.write("Sugestão: Você está realizando um estudo sobre a proporção de clientes satisfeitos em relação a um novo serviço de entrega. Você gostaria de estimar a proporção de clientes satisfeitos com um nível de confiança de 90% e uma margem de erro de 5%. Supondo que você não tenha uma estimativa inicial da proporção de clientes satisfeitos, quantos clientes você precisa entrevistar para obter uma estimativa razoável?")

    margem_erro = st.number_input('Margem de Erro (%) :', min_value=0.0, max_value=100.0, value=1.0)
    conf = st.number_input('Valor de Confiança (%) :', min_value=0.0, max_value=100.0, value=95.0)

    # Comentários

    z = abs(norm.ppf((1 + (conf / 100)) / 2))
    tamanho_amostra = ((z**2 * 0.5 * 0.5) / (margem_erro / 100)**2)

    st.write(f'Tamanho da Amostra para Proporção: {tamanho_amostra:.2f}')

def hipotese_media():
    st.title('Teste de Hipótese para Média')
    st.write("Exercício: Conduza um teste de hipótese para verificar se a média populacional de uma determinada característica é igual a um valor específico, com base em uma amostra.")

    # Inputs do usuário
    media_amostra = st.number_input("Média da amostra:", min_value=-1000.0, max_value=1000.0, value=1.0, format="%.6f")
    media_populacional = st.number_input("Média Populacional (H0):", min_value=-1000.0, max_value=1000.0, value=1.0, format="%.6f")
    desvio_padrao_populacao = st.number_input("Desvio Padrão da População:", min_value=0.1, max_value=1000.0, value=1.0, format="%.6f")
    tamanho_amostra = st.number_input("Tamanho da amostra:", min_value=1, max_value=1000, value=30)
    valor_critico_z = st.number_input("Valor crítico Z:", min_value=-6.0, max_value=6.0, value=1.0, format="%.6f")

    
    # Comentários
    

    
    # Cálculos
    z = (media_amostra - media_populacional) / (desvio_padrao_populacao / np.sqrt(tamanho_amostra))
    p_value = 2 * (1 - norm.cdf(abs(z)))
    decisao = "Rejeita H0" if abs(z) > abs(valor_critico_z) else "Aceita H0"

    # Exibição dos resultados
    st.text(f"Valor Z: {z:.6f}")
    st.text(f"P-valor: {p_value:.6f}")
    st.text(f"Decisão: {decisao}")

# Função para a Regressão Linear Simples
def regressao_linear_simples():
    st.title('Regressão Linear Simples')
    st.write("Exercício: Crie um modelo de regressão linear simples para prever uma variável dependente com base em uma variável independente.")

    # Inputs do usuário
    x_values = st.text_input("Valores de x (separados por vírgula):", "1,2,3,4,5,6,7,8,9,10")
    y_values = st.text_input("Valores de y (separados por vírgula):", "3,7,8,12,14,15,20,22,25,30")
    x_previsao = st.number_input("Valor de x para previsão:", min_value=-100.0, max_value=100.0, value=1.0, format="%.6f")

    # Comentários

    

    # Convertendo os valores de entrada
    x = np.array([float(i) for i in x_values.split(',')])
    y = np.array([float(i) for i in y_values.split(',')])

    # Ajustando o modelo linear
    modelo = LinearRegression().fit(x.reshape(-1, 1), y)
    previsto = modelo.predict(np.array([[x_previsao]]))[0]

    # Exibição da previsão
    st.text(f"Previsão para x = {x_previsao}: {previsto:.2f}")

    # Plotando os dados e a linha de regressão
    plt.figure(figsize=(10, 6))
    plt.scatter(x, y, color='blue', label='Dados')
    plt.plot(x, modelo.predict(x.reshape(-1, 1)), color='red', label='Linha de Regressão')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Regressão Linear Simples')
    plt.legend()
    st.pyplot(plt)
