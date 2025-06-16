import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import os
import sys

# Importa a função que queremos testar
from advanced_pca_simplificado import (
    display_interactive_pca_example
)

# Configuração básica do Streamlit
st.set_page_config(
    page_title="Validação PCA",
    page_icon="🧪",
    layout="wide"
)

st.title("🧪 Validação da Função display_interactive_pca_example")

# Cria dados de teste simples
np.random.seed(42)
n_samples = 100
n_features = 5

# Dados sintéticos
data = np.random.randn(n_samples, n_features)
selected_assets = ["Asset_" + str(i) for i in range(1, n_features+1)]

# Aplica PCA
pca = PCA(n_components=3)
pca.fit(data)

# Exibe informações sobre os dados de teste
st.subheader("Dados de teste")
st.write(f"Número de amostras: {n_samples}")
st.write(f"Número de ativos: {n_features}")
st.write(f"Componentes PCA: 3")

st.subheader("Variância explicada por componente")
explained_variance = pd.DataFrame({
    "Componente": [f"PC{i+1}" for i in range(3)],
    "Variância Explicada": pca.explained_variance_ratio_
})
st.table(explained_variance)

# Linha divisória
st.markdown("---")

st.subheader("Resultado da função display_interactive_pca_example:")
try:
    # Tenta executar a função
    display_interactive_pca_example(pca, selected_assets, 3)
    st.success("✅ A função foi executada com sucesso!")
except Exception as e:
    st.error(f"❌ Erro ao executar a função: {str(e)}")
    st.code(str(e))
    
    # Exibe o traceback completo
    import traceback
    st.code(traceback.format_exc())
