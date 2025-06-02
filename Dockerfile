# Dockerfile para executar a aplicação Streamlit
FROM python:3.9-slim

WORKDIR /app

# Copia e instala dependências
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo o código
COPY . .

# Expor porta do Streamlit
EXPOSE 8501

# Comando padrão
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
