# Use uma imagem base com Python
FROM python:3.12-slim

# Instale as dependências do sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Instale as dependências do Python
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copie o código do aplicativo para o contêiner
COPY . /app
WORKDIR /app

# Defina o comando para iniciar o aplicativo
CMD ["streamlit", "run", "webapp.py"]
