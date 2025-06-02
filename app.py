import streamlit as st
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import yfinance as yf
import os
import socket
import itertools  # para combinação de portfólios

# Exibe logo da ferramenta (homenagem a Khaio Geovan)
logo = Image.open('logo.png')
st.image(logo, width=150)

# Configurações
DATA_DIR = 'data'
RAW_DATA = os.path.join(DATA_DIR, 'raw_data.csv')

st.title('Análise de Portfólio com PCA')
st.markdown(
    'Bem-vindo! Esta ferramenta permite analisar carteiras de investimento de forma simples. '  
    'Siga os passos para baixar dados, selecionar ativos, avaliar desempenho e gerar exportações.'
)
st.markdown(
    '[Acesse online e compartilhe ▶️](https://bilionario-3w62sdcxhsf3i8yfqywoaq.streamlit.app/)'
)
# Botão para baixar dados
if st.button('Baixar dados dos ativos'):
    st.markdown('**Passo 1:** Baixe dados históricos ajustados de preços via Yahoo Finance.')
    from data_fetch import fetch_data
    fetch_data()
    st.success('Dados baixados com sucesso!')

# Carregar dados se existir
if os.path.exists(RAW_DATA):
    df = pd.read_csv(RAW_DATA, index_col=0, parse_dates=True)
    # Mantém dataframe completo para cálculos de melhor combinação
    df_full = df.copy()
    # Calcula retornos completos para auto-seleção
    returns_full = df_full / df_full.shift(1) - 1
    returns_full = returns_full.dropna()
    monthly_full = (1 + returns_full).resample('ME').prod() - 1
    # Botão para auto-preencher seleção de ativos
    if 'selected' not in st.session_state:
        st.session_state['selected'] = df_full.columns.tolist()[:5]
    if st.sidebar.button('Auto-preencher melhores ativos'):
        best_avg = -float('inf')
        best_combo = None
        # Testa combinações de 3 a 10
        for k in range(3, min(10, len(df_full.columns)) + 1):
            for combo in itertools.combinations(df_full.columns, k):
                avg = monthly_full[list(combo)].mean(axis=1).mean()
                if avg > best_avg:
                    best_avg = avg
                    best_combo = combo
        st.session_state['selected'] = list(best_combo)
        st.sidebar.success(f'Selecionados: {best_combo}')
    # Multiselect de ativos com session_state
    selected = st.sidebar.multiselect(
        'Selecione ativos (3-10)', df.columns.tolist(),
        default=st.session_state['selected'],
        key='selected'
    )
    if len(selected) < 3 or len(selected) > 10:
        st.warning('Selecione entre 3 e 10 ativos para prosseguir')
        st.stop()
    df = df[selected]

    # Cálculo de retornos (diferença percentual manual para evitar warnings)
    returns = df / df.shift(1) - 1
    returns = returns.dropna()
    # Suprimir warnings de FutureWarning
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning)

    # Seção de Performance do Portfólio
    st.markdown('**Passo 3:** Veja as métricas de desempenho do seu portfólio, como retorno total, volatilidade e drawdown.')
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

    # Botão para gerar portfólio ótimo baseado em top 3 ativos por retorno mensal médio
    st.markdown('**Passo 5:** Encontre a combinação ótima de ativos com melhor retorno mensal médio.')
    if st.button('Gerar Portfólio Ótimo (Top 3 Retorno Mensal)'):
        # Retornos mensais de cada ativo
        monthly_ret = (1 + returns).resample('M').prod() - 1
        # Média de retorno mensal por ativo
        avg_mon = monthly_ret.mean().sort_values(ascending=False)
        top3 = avg_mon.head(3).index.tolist()
        # Formata os valores de retorno médio mensal de cada ativo
        avg_values = avg_mon.head(3).values * 100
        fmt_vals = ', '.join([f"{v:.2f}%" for v in avg_values])
        # Exibe Top 3 ativos com retornos corretamente formatados
        st.markdown(f"**Top 3 Ativos:** {', '.join(top3)} com retorno mensal médio de {fmt_vals}")
        # Cálculo para portfólio top3
        returns_top3 = returns[top3]
        portf_top3 = returns_top3.mean(axis=1)
        cum_top3 = (1 + portf_top3).cumprod() * initial_capital
        st.subheader('Valor Cumulado - Portfólio Ótimo Top 3')
        st.line_chart(cum_top3, height=200)
        # Mostrar métricas do portfólio ótimo
        tr_opt = cum_top3.iloc[-1]/initial_capital - 1
        ar_opt = portf_top3.mean()*252
        av_opt = portf_top3.std()*(252**0.5)
        dd_opt = ((cum_top3 - cum_top3.cummax())/cum_top3.cummax()).min()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric('Retorno Total (Top3)', f"{tr_opt:.2%}")
        c2.metric('Retorno Anualizado (Top3)', f"{ar_opt:.2%}")
        c3.metric('Volatilidade Anual (Top3)', f"{av_opt:.2%}")
        c4.metric('Drawdown Máximo (Top3)', f"{dd_opt:.2%}")

    # Retorno Mensal do Portfólio
    st.markdown('**Passo 4:** Acompanhe o retorno de cada mês para entender a consistência dos ganhos.')
    st.subheader('Retorno Mensal do Portfólio')
    # retorna retornos mensais do portfólio e de cada ativo
    monthly_portfolio = (1 + portfolio_returns).resample('ME').prod() - 1
    st.bar_chart(monthly_portfolio)
    # DataFrame de retornos mensais por ativo (para encontrar melhor combinação)
    monthly_assets = (1 + returns).resample('ME').prod() - 1

    # Melhor portfólio baseado em retorno médio mensal (ativos)
    if st.button('Melhor Portfólio Mensal'):
        best_avg = -float('inf')
        best_combo = None
        best_series = None
        # busca combinatória de 3 até todos ativos selecionados
        for k in range(3, len(selected)+1):
            for combo in itertools.combinations(selected, k):
                avg = monthly_assets[list(combo)].mean(axis=1).mean()
                if avg > best_avg:
                    best_avg = avg
                    best_combo = combo
                    best_series = monthly_assets[list(combo)].mean(axis=1)
        # Exibir resultados
        st.subheader('Melhor Portfólio (Retorno Mensal)')
        st.write(f"Ativos: {best_combo}")
        st.write(f"Retorno médio mensal: {best_avg:.2%}")
        st.line_chart(best_series)

    # filtro por retorno mínimo do portfólio
    min_month = st.slider('Retorno Mensal Mínimo (%)', -20.0, 20.0, 0.0)
    highlight = monthly_portfolio[monthly_portfolio >= (min_month/100)]
    if not highlight.empty:
        st.markdown(f"**Meses com retorno mensal do portfólio ≥ {min_month:.2f}%**")
        st.write(highlight.apply(lambda x: f"{x:.2%}"))

    # PCA
    st.markdown('**Passo 6:** Analise a estrutura de risco usando PCA: variância explicada, scree plot e cargas dos componentes.')
    # PCA: número de componentes não pode exceder ativos; default em 5 ou menos
    n_feats = returns.shape[1]
    default_n = min(5, n_feats)
    n_components = st.sidebar.slider(
        'Número de componentes PCA', 1, n_feats, default_n
    )

    # PCA
    pca = PCA(n_components=n_components, random_state=42)
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
    st.markdown('**Passo 7:** Exporte preços e parâmetros de indicadores técnicos para uso no StrategyQuant.')
    if st.button('Exportar para StrategyQuant'): 
        from export_strategyquant import export_to_excel
        export_to_excel()
        st.success('Arquivo strategyquant_data.xlsx gerado!')
else:
    st.info('Clique em "Baixar dados dos ativos" para iniciar.')
