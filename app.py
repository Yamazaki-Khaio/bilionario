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

    # Cálculo de retornos
    returns = df.pct_change().dropna()
    
    # Seção de Performance do Portfólio
    st.subheader('Performance do Portfólio')
    initial_capital = st.number_input('Capital Inicial (R$)', min_value=100.0, value=10000.0, step=100.0)
    # Retorno do portfólio igualmente ponderado
    portfolio_returns = returns.mean(axis=1)
    portfolio_cum = (1 + portfolio_returns).cumprod() * initial_capital
    # Métricas
    total_return = portfolio_cum.iloc[-1] / initial_capital - 1
    annual_return = portfolio_returns.mean() * 252
    annual_vol = portfolio_returns.std() * (252**0.5)
    running_max = portfolio_cum.cummax()
    drawdown = (portfolio_cum - running_max) / running_max
    max_dd = drawdown.min()
    # Exibição de métricas
    col1, col2, col3, col4 = st.columns(4)
    col1.metric('Retorno Total', f"{total_return:.2%}")
    col2.metric('Retorno Anualizado', f"{annual_return:.2%}")
    col3.metric('Volatilidade Anual', f"{annual_vol:.2%}")
    col4.metric('Drawdown Máximo', f"{max_dd:.2%}")
    # Gráfico de valor cumulativo
    st.line_chart(portfolio_cum, height=250)
    # Risco vs Retorno
    import plotly.express as px
    fig_rr = px.scatter(x=[annual_vol], y=[annual_return], text=['Portfólio'], labels={'x':'Volatilidade Anual','y':'Retorno Anualizado'}, title='Risco vs Retorno')
    fig_rr.update_traces(textposition='top center')
    st.plotly_chart(fig_rr)

    # Retorno Mensal do Portfólio
    st.subheader('Retorno Mensal do Portfólio')
    # agrega retornos para cada mês
    monthly_returns = (1 + portfolio_returns).resample('M').prod() - 1
    st.bar_chart(monthly_returns)
    # filtro por retorno mínimo
    min_month = st.slider('Retorno Mensal Mínimo (%)', -20.0, 20.0, 0.0)
    highlight = monthly_returns[monthly_returns >= (min_month/100)]
    if not highlight.empty:
        st.markdown(f"**Meses com retorno mensal ≥ {min_month:.2f}%**")
        st.write(highlight.apply(lambda x: f"{x:.2%}"))

    # PCA
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
