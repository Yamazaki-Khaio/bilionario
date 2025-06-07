import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import warnings
import itertools
from PIL import Image

warnings.filterwarnings("ignore")

# Imports dos módulos customizados
from mt5_parser import MT5ReportParser
from pca_advanced import PCAAdvancedAnalysis
from pair_trading import PairTradingAnalysis
from portfolio_allocation import PortfolioAllocationManager
from data_fetch import ASSET_CATEGORIES

# Constantes
MT5_REAL_LABEL = 'MT5 Real'
PCA_PORTFOLIO_LABEL = 'PCA Portfolio'
SYMBOL_COLUMN = 'Símbolo'
PL_ABS_COLUMN = 'P&L_Abs'
LOSS_LABEL = 'Prejuízo'
PROFIT_LABEL = 'Lucro'
TOTAL_RETURN_LABEL = "Retorno Total"
MAX_DRAWDOWN_LABEL = "Max Drawdown"

# =====================================================================
# CONFIGURAÇÃO DA PÁGINA
# =====================================================================

st.set_page_config(
    page_title="💰 Análise Bilionário",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================================
# FUNÇÕES AUXILIARES
# =====================================================================

def calculate_metrics(returns, initial_capital):
    """Calcula métricas de performance do portfolio"""
    if len(returns) == 0:
        return {
            'total_return': 0.0,
            'annual_return': 0.0,
            'annual_volatility': 0.0,
            'max_drawdown': 0.0
        }
    
    equity_curve = (1 + returns).cumprod() * initial_capital
    total_return = (equity_curve.iloc[-1] / initial_capital) - 1
    
    n_periods = len(returns)
    if n_periods > 0:
        annual_return = (1 + total_return) ** (252 / n_periods) - 1
    else:
        annual_return = 0.0
    
    annual_volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0.0
    
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve / running_max - 1)
    max_drawdown = drawdown.min() if len(drawdown) > 0 else 0.0
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'annual_volatility': annual_volatility,
        'max_drawdown': max_drawdown,
        'equity_curve': equity_curve,
        'drawdown': drawdown
    }

def get_monthly_returns(returns):
    """Calcula retornos mensais"""
    monthly = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
    return monthly

def load_mt5_data():
    """Carrega dados MT5 do sidebar"""
    uploaded_mt5 = st.sidebar.file_uploader(
        "📊 Upload relatório MT5 (HTML)", 
        type=['html', 'htm'],
        help="Faça upload do relatório HTML exportado do MetaTrader 5"
    )

    mt5_data = None
    if uploaded_mt5 is not None:
        try:
            file_type = '.html' if uploaded_mt5.name.lower().endswith(('.html', '.htm')) else '.pdf'
            parser = MT5ReportParser(uploaded_mt5, file_type)
            mt5_data = parser.get_portfolio_summary()
            
            st.sidebar.success("✅ MT5 carregado com sucesso!")
            st.sidebar.write(f"**Conta:** {mt5_data['account_name']}")
            st.sidebar.write(f"**Saldo:** R$ {mt5_data['balance']:,.2f}")
            st.sidebar.write(f"**Lucro:** R$ {mt5_data['net_profit']:,.2f}")
            st.sidebar.write(f"**Retorno:** {mt5_data['gain']}")
            
        except Exception as e:
            st.sidebar.error(f"❌ Erro ao processar MT5: {str(e)}")
            mt5_data = None
    
    return mt5_data

# =====================================================================
# PÁGINAS DA APLICAÇÃO
# =====================================================================

def show_home_page():
    """Página inicial"""
    # Logo
    try:
        logo = Image.open('logo.png')
        st.image(logo, width=150)
    except FileNotFoundError:
        st.write("🚀 **Análise de Portfólio - by Khaio Geovan**")

    st.title("🚀 Bilionário - Análise de Portfolio com PCA")
    st.markdown('Bem-vindo! Ferramenta de análise de carteiras.')
    st.markdown('[Acesse online ▶️](https://bilionario-3w62sdcxhsf3i8yfqywoaq.streamlit.app/)')

    # Informações sobre a aplicação
    st.markdown("---")
    st.subheader("ℹ️ Sobre a Aplicação")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **🎯 Funcionalidades:**
        - Análise PCA de portfolios
        - Comparação com dados MT5
        - Gestão por setores
        - Análise avançada PCA
        - Pair Trading
        """)
    
    with col2:
        st.markdown("""
        **📊 Métricas Calculadas:**
        - Retorno Total e Anualizado
        - Volatilidade
        - Sharpe Ratio
        - Max Drawdown
        - Índices de risco
        """)

    # Download de dados
    st.markdown("---")
    st.subheader("📥 Dados dos Ativos")
    
    if st.button('🔄 Baixar/Atualizar dados dos ativos'):
        with st.spinner('Baixando dados...'):
            from data_fetch import fetch_data
            fetch_data()
        st.success('✅ Dados baixados com sucesso!')

    # Status dos dados
    DATA_DIR = 'data'
    RAW_DATA = os.path.join(DATA_DIR, 'raw_data.csv')
    
    if os.path.exists(RAW_DATA):
        df = pd.read_csv(RAW_DATA, index_col=0, parse_dates=True)
        st.success(f"📊 Dados disponíveis: {len(df.columns)} ativos, {len(df)} dias")
        
        # Preview dos dados
        with st.expander("👁️ Preview dos Dados"):
            st.dataframe(df.tail(10), use_container_width=True)
            
        # Estatísticas básicas
        with st.expander("📈 Estatísticas Básicas"):
            returns = df.pct_change().dropna()
            stats = pd.DataFrame({
                'Retorno Médio (%)': (returns.mean() * 100).round(3),
                'Volatilidade (%)': (returns.std() * 100).round(3),
                'Sharpe (aprox)': (returns.mean() / returns.std()).round(3)
            })
            st.dataframe(stats, use_container_width=True)
    else:
        st.warning("⚠️ Dados não encontrados. Clique no botão acima para baixar os dados.")

def show_pca_performance_page():
    """Página de análise PCA"""
    st.title("📊 Performance PCA")
    
    DATA_DIR = 'data'
    RAW_DATA = os.path.join(DATA_DIR, 'raw_data.csv')
    
    if not os.path.exists(RAW_DATA):
        st.error("❌ Dados não encontrados. Vá para a página Home e baixe os dados primeiro.")
        return
    
    df = pd.read_csv(RAW_DATA, index_col=0, parse_dates=True)
    returns = df.pct_change().dropna()
    
    # Seleção de ativos
    st.sidebar.subheader("🎯 Seleção de Ativos")
    
    if 'selected' not in st.session_state:
        st.session_state['selected'] = df.columns.tolist()[:5]
    
    # Auto-seleção
    if st.sidebar.button('🎲 Auto-seleção'):
        monthly = get_monthly_returns(returns)
        best, combo = -1, None
        for k in range(3, min(10, len(df.columns)) + 1):
            for c in itertools.combinations(df.columns, k):
                avg = monthly[list(c)].mean(axis=1).mean()
                if avg > best: 
                    best, combo = avg, c
        st.session_state['selected'] = list(combo)
        st.sidebar.success(f"✅ Selecionados: {len(combo)} ativos")
        
    selected = st.sidebar.multiselect(
        'Selecione ativos (3-20)', 
        df.columns.tolist(),
        default=st.session_state['selected']
    )
    
    if not 3 <= len(selected) <= 20:
        st.warning('⚠️ Selecione entre 3 e 20 ativos para análise PCA')
        return
        
    df_selected = df[selected]
    returns_selected = returns[selected]
    
    # Parâmetros
    initial_capital = st.sidebar.number_input(
        '💰 Capital Inicial (R$)', 
        min_value=100.0, 
        max_value=1e7, 
        value=10000.0, 
        step=100.0
    )
    
    # Cálculo da performance
    if returns_selected.empty:
        st.warning("⚠️ Não há dados suficientes para análise.")
        return
    
    portf_ret = returns_selected.mean(axis=1)
    portf_cum = (1 + portf_ret).cumprod() * initial_capital
    metrics = calculate_metrics(portf_ret, initial_capital)
    
    # Métricas principais
    st.subheader("📊 Métricas de Performance")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(TOTAL_RETURN_LABEL, f"{metrics['total_return']:.2%}")
    with col2:
        st.metric("Retorno Anualizado", f"{metrics['annual_return']:.2%}")
    with col3:
        st.metric("Volatilidade", f"{metrics['annual_volatility']:.2%}")
    with col4:
        st.metric(MAX_DRAWDOWN_LABEL, f"{metrics['max_drawdown']:.2%}")
    
    # Gráfico de evolução
    st.subheader("📈 Evolução do Portfolio")
    fig_evolution = px.line(
        x=portf_cum.index, 
        y=portf_cum.values,
        title='Evolução do Capital',
        labels={'x': 'Data', 'y': 'Capital (R$)'}
    )
    st.plotly_chart(fig_evolution, use_container_width=True)
    
    # Análise PCA
    st.subheader("🔍 Análise PCA")
    
    n_components = st.sidebar.slider(
        'Número de Componentes PCA', 
        min_value=1, 
        max_value=len(selected), 
        value=min(5, len(selected))
    )
    
    # Aplicar PCA
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(returns_selected.fillna(0))
    pca = PCA(n_components=n_components, random_state=42)
    components = pca.fit_transform(scaled_data)
    
    # Variância explicada
    explained_var = pca.explained_variance_ratio_
    fig_scree = px.bar(
        x=list(range(1, len(explained_var) + 1)), 
        y=explained_var * 100,
        title='Scree Plot - Variância Explicada por Componente',
        labels={'x': 'Componente', 'y': 'Variância Explicada (%)'}
    )
    st.plotly_chart(fig_scree, use_container_width=True)
    
    # Scatter plot dos primeiros 2 componentes
    if n_components >= 2:
        fig_scatter = px.scatter(
            x=components[:, 0], 
            y=components[:, 1],
            title='Primeiros 2 Componentes Principais',
            labels={'x': 'PC1', 'y': 'PC2'}
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Retornos mensais
    st.subheader("📅 Retornos Mensais")
    monthly_returns = get_monthly_returns(portf_ret)
    fig_monthly = px.bar(
        x=monthly_returns.index, 
        y=monthly_returns.values,
        title='Retornos Mensais do Portfolio'
    )
    st.plotly_chart(fig_monthly, use_container_width=True)

def show_mt5_comparison_page():
    """Página de comparação com MT5"""
    st.title("⚖️ Comparação PCA vs MT5")
    
    # Verificar se há dados MT5
    mt5_data = st.session_state.get('mt5_data')
    if not mt5_data:
        st.warning("⚠️ Nenhum dado MT5 carregado. Faça upload de um relatório MT5 no sidebar.")
        return
    
    st.success("✅ Dados MT5 carregados com sucesso!")
    
    # Exibir resumo MT5
    st.subheader("📊 Resumo MT5")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Saldo Atual", f"R$ {mt5_data.get('balance', 0):,.2f}")
    with col2:
        st.metric("Lucro Líquido", f"R$ {mt5_data.get('net_profit', 0):,.2f}")
    with col3:
        st.metric("Ganho", mt5_data.get('gain', 'N/A'))
    with col4:
        st.metric("Drawdown", mt5_data.get('drawdown', 'N/A'))
    
    # Comparação com PCA (se dados disponíveis)
    DATA_DIR = 'data'
    RAW_DATA = os.path.join(DATA_DIR, 'raw_data.csv')
    
    if os.path.exists(RAW_DATA):
        st.subheader("📈 Comparação PCA vs MT5")
        st.info("💡 Esta seção compara a performance do portfolio PCA com os resultados reais do MT5")
        
        # Aqui você pode implementar gráficos de comparação
        # Por exemplo, comparar retornos, drawdowns, etc.
        
    else:
        st.warning("⚠️ Dados PCA não disponíveis para comparação.")

def show_sector_management_page():
    """Página de gestão por setor"""
    st.title("💰 Gestão por Setor")
    
    DATA_DIR = 'data'
    RAW_DATA = os.path.join(DATA_DIR, 'raw_data.csv')
    
    if not os.path.exists(RAW_DATA):
        st.error("❌ Dados não encontrados. Vá para a página Home e baixe os dados primeiro.")
        return
    
    df = pd.read_csv(RAW_DATA, index_col=0, parse_dates=True)
    returns = df.pct_change().dropna()
    
    st.subheader("🏭 Alocação por Setores")
    st.info("💡 Configure a alocação de capital por setor da economia")
    
    # Setores disponíveis (baseado nas categorias do data_fetch)
    available_sectors = list(ASSET_CATEGORIES.keys())
    
    # Interface para configurar alocação por setor
    st.subheader("⚙️ Configuração de Alocação")
    
    sector_allocations = {}
    total_allocation = 0
    
    for sector in available_sectors:
        allocation = st.slider(
            f"Alocação {sector} (%)",
            min_value=0,
            max_value=100,
            value=20,  # Default
            key=f"allocation_{sector}"
        )
        sector_allocations[sector] = allocation
        total_allocation += allocation
    
    if total_allocation != 100:
        st.warning(f"⚠️ A alocação total é {total_allocation}%. Ajuste para 100%.")
    else:
        st.success("✅ Alocação balanceada!")
        
        # Mostrar alocação em gráfico
        fig_allocation = px.pie(
            values=list(sector_allocations.values()),
            names=list(sector_allocations.keys()),
            title='Alocação por Setor'
        )
        st.plotly_chart(fig_allocation, use_container_width=True)

def show_advanced_pca_page():
    """Página de PCA avançado"""
    st.title("🔬 PCA Avançado")
    
    DATA_DIR = 'data'
    RAW_DATA = os.path.join(DATA_DIR, 'raw_data.csv')
    
    if not os.path.exists(RAW_DATA):
        st.error("❌ Dados não encontrados. Vá para a página Home e baixe os dados primeiro.")
        return
    
    try:
        df = pd.read_csv(RAW_DATA, index_col=0, parse_dates=True)
        returns = df.pct_change().dropna()
        
        st.subheader("🧠 Análise PCA Avançada")
        st.info("💡 Análise profunda usando PCA com recursos avançados")
        
        # Seleção de ativos
        selected_assets = st.multiselect(
            "Selecione ativos para análise avançada:",
            df.columns.tolist(),
            default=df.columns.tolist()[:10]
        )
        
        if len(selected_assets) >= 3:
            # Usar módulo PCA avançado
            pca_advanced = PCAAdvancedAnalysis(df[selected_assets])
            
            # Parâmetros avançados
            col1, col2 = st.columns(2)
            with col1:
                window_size = st.slider("Janela de análise (dias)", 30, 252, 90)
            with col2:
                confidence_level = st.slider("Nível de confiança (%)", 90, 99, 95)
            
            # Executar análise
            if st.button("🚀 Executar Análise Avançada"):
                with st.spinner("Processando análise avançada..."):
                    try:
                        results = pca_advanced.run_advanced_analysis(
                            window_size=window_size,
                            confidence_level=confidence_level/100
                        )
                        
                        if results:
                            st.success("✅ Análise concluída!")
                            # Aqui você exibiria os resultados
                            st.json(results)  # Placeholder
                        else:
                            st.error("❌ Erro na análise avançada")
                            
                    except Exception as e:
                        st.error(f"❌ Erro: {str(e)}")
        else:
            st.warning("⚠️ Selecione pelo menos 3 ativos para análise avançada")
            
    except Exception as e:
        st.error(f"❌ Erro ao carregar dados: {str(e)}")

def show_pair_trading_page():
    """Página de pair trading"""
    st.title("🔄 Pair Trading")
    
    DATA_DIR = 'data'
    RAW_DATA = os.path.join(DATA_DIR, 'raw_data.csv')
    
    if not os.path.exists(RAW_DATA):
        st.error("❌ Dados não encontrados. Vá para a página Home e baixe os dados primeiro.")
        return
    
    try:
        df = pd.read_csv(RAW_DATA, index_col=0, parse_dates=True)
        
        st.subheader("👫 Análise de Pair Trading")
        st.info("💡 Identifique oportunidades de arbitragem entre pares de ativos")
        
        # Seleção de pares
        col1, col2 = st.columns(2)
        
        with col1:
            asset1 = st.selectbox("Primeiro ativo:", df.columns.tolist())
        with col2:
            asset2 = st.selectbox("Segundo ativo:", df.columns.tolist())
        
        if asset1 != asset2:
            # Usar módulo de pair trading
            pair_analyzer = PairTradingAnalysis(df[[asset1, asset2]])
            
            # Parâmetros
            lookback_period = st.slider("Período de lookback (dias)", 20, 252, 60)
            z_threshold = st.slider("Z-Score threshold", 1.0, 3.0, 2.0, 0.1)
            
            # Executar análise
            if st.button("📊 Analisar Par"):
                with st.spinner("Analisando pair trading..."):
                    try:
                        analysis = pair_analyzer.analyze_pair(
                            asset1, asset2,
                            lookback_period=lookback_period,
                            z_threshold=z_threshold
                        )
                        
                        if analysis:
                            st.success("✅ Análise de par concluída!")
                            
                            # Métricas do par
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Correlação", f"{analysis.get('correlation', 0):.3f}")
                            with col2:
                                st.metric("Cointegração", analysis.get('cointegration', 'N/A'))
                            with col3:
                                st.metric("Oportunidades", analysis.get('signals', 0))
                                
                            # Aqui você adicionaria gráficos e mais detalhes
                            
                        else:
                            st.error("❌ Erro na análise do par")
                            
                    except Exception as e:
                        st.error(f"❌ Erro: {str(e)}")
        else:
            st.warning("⚠️ Selecione dois ativos diferentes")
            
    except Exception as e:
        st.error(f"❌ Erro ao carregar dados: {str(e)}")

# =====================================================================
# NAVEGAÇÃO PRINCIPAL
# =====================================================================

def main():
    """Função principal da aplicação"""
    
    # Sistema de navegação por sidebar
    st.sidebar.title("🧭 Navegação")
    page = st.sidebar.selectbox(
        "Escolha a seção:",
        [
            "🏠 Home",
            "📊 Performance PCA", 
            "⚖️ Comparação MT5",
            "💰 Gestão por Setor",
            "🔬 PCA Avançado", 
            "🔄 Pair Trading"
        ]
    )
    
    st.sidebar.markdown("---")
    
    # Carregar dados MT5 e armazenar no session_state
    mt5_data = load_mt5_data()
    if mt5_data:
        st.session_state['mt5_data'] = mt5_data
    
    # Roteamento das páginas
    if page == "🏠 Home":
        show_home_page()
    elif page == "📊 Performance PCA":
        show_pca_performance_page()
    elif page == "⚖️ Comparação MT5":
        show_mt5_comparison_page()
    elif page == "💰 Gestão por Setor":
        show_sector_management_page()
    elif page == "🔬 PCA Avançado":
        show_advanced_pca_page()
    elif page == "🔄 Pair Trading":
        show_pair_trading_page()

# Executar aplicação
if __name__ == "__main__":
    main()
