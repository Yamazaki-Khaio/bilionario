import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import yfinance as yf
import os

# Configurações
DATA_DIR = 'data'
RAW_DATA = os.path.join(DATA_DIR, 'raw_data.csv')

st.title('Análise de Portfólio com PCA')

# Botão para baixar dados
if st.button('Baixar dados dos ativos'):
    from data_fetch import fetch_data
    fetch_data()
    st.success('Dados baixados com sucesso!')

# Carregar dados se existir
if os.path.exists(RAW_DATA):
    df = pd.read_csv(RAW_DATA, index_col=0, parse_dates=True)
    st.subheader('Preços Ajustados')
    st.dataframe(df.tail())

    # Retornos
    returns = df.pct_change().dropna()
    n_components = st.sidebar.slider('Número de componentes PCA', 1, min(returns.shape[1], 10), 5)

    # PCA
    pca = PCA(n_components=n_components)
    components = pca.fit_transform(returns)
    explained = pca.explained_variance_ratio_

    # Variância explicada
    st.subheader('Variância Explicada por Componente')
    fig, ax = plt.subplots()
    ax.bar(range(1, n_components+1), explained)
    ax.set_xlabel('Componentes')
    ax.set_ylabel('Variância Explicada')
    st.pyplot(fig)

    # Scatter dos primeiros 2 componentes
    if n_components >= 2:
        st.subheader('Scatter dos 2 Primeiros Componentes')
        fig2, ax2 = plt.subplots()
        ax2.scatter(components[:,0], components[:,1], alpha=0.5)
        ax2.set_xlabel('Componente 1')
        ax2.set_ylabel('Componente 2')
        st.pyplot(fig2)

    # Exportar para Excel StrategyQuant
    if st.button('Exportar para StrategyQuant'): 
        from export_strategyquant import export_to_excel
        export_to_excel()
        st.success('Arquivo strategyquant_data.xlsx gerado!')
else:
    st.info('Clique em "Baixar dados dos ativos" para iniciar.')
