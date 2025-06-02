# Dockerfile para executar a aplicação Streamlit
FROM python:3.9-slim

WORKDIR /app

# Instala pacotes de sistema necessários
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    software-properties-common \
  && rm -rf /var/lib/apt/lists/*

# Copia e instala dependências Python
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo o código da aplicação
COPY . .

# Expor porta padrão do Streamlit
EXPOSE 8501

# Healthcheck para verificar se o Streamlit está rodando
HEALTHCHECK --interval=30s --timeout=5s CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Entrypoint padrão
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
