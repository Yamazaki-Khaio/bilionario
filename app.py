import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import warnings
import itertools
import time
from PIL import Image

warnings.filterwarnings("ignore")

# Imports dos módulos customizados
from constants import (
    FINAL_CAPITAL_LABEL, TOTAL_RETURN_LABEL, METRIC_LABEL,
    PETR4_SYMBOL, LOADING_PC1_LABEL, LOADING_PC2_LABEL,
    WEIGHT_PERCENTAGE_LABEL, MT5_REAL_LABEL, PCA_PORTFOLIO_LABEL,
    TELECOMMUNICATIONS_SECTOR, MATERIALS_SECTOR, TECHNOLOGY_SECTOR,
    MAX_DRAWDOWN_LABEL, ANNUAL_RETURN_LABEL, SHARPE_RATIO_LABEL,
    RESULTS_INTERPRETATION_LABEL, CORRELATION_LABEL,
    PCA_PERFORMANCE_TITLE, MT5_COMPARISON_TITLE, ADVANCED_PCA_TITLE, 
    PAIR_TRADING_TITLE, STATISTICAL_ANALYSIS_TITLE, NORMALITY_ANALYSIS_TITLE,
    DATA_NOT_FOUND_MSG, SELECT_VALID_PAIR_MSG,
    get_raw_data_path, DATA_DIR, RAW_DATA_FILENAME,
    SYMBOL_COLUMN, PL_ABS_COLUMN, LOSS_LABEL, PROFIT_LABEL
)
from mt5_parser import MT5ReportParser
from trading_report_parser import TradingReportParser  # Novo parser
from pca_advanced import PCAAdvancedAnalysis
from pair_trading import PairTradingAnalysis
from portfolio_allocation import PortfolioAllocationManager
from statistical_analysis import StatisticalAnalysis
from pair_trading_advanced import PairTradingAdvanced
from trading_report_parser import TradingReportParser
from mt5_comparison_helpers import (
    validate_mt5_data, validate_pca_data, setup_mt5_comparison_sidebar,
    select_pca_assets, calculate_pca_metrics, display_pca_summary,
    display_mt5_summary, display_comparative_analysis, display_recommendations,
    plot_comparative_performance, analyze_symbol_performance, create_risk_metrics_radar
)
from data_fetch import ASSET_CATEGORIES
from financial_formatting import (
    format_percentage, format_currency, format_ratio, 
    auto_format_metric, FINANCIAL_METRICS_FORMAT
)
from app_helpers import (
    plot_temporal_comparison, plot_drawdown_comparison,
    create_risk_metrics_analysis, create_performance_radar_chart,
    create_portfolio_allocation_analysis
)
from pdf_export_helpers import create_download_button, generate_complete_statistical_analysis_pdf

# =====================================================================
# CONFIGURAÇÃO DA PÁGINA
# =====================================================================

st.set_page_config(
    page_title="💰 Análise Bilionário",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="collapsed"
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
            'monthly_return': 0.0,
            'annual_volatility': 0.0,
            'max_drawdown': 0.0
        }
    
    equity_curve = (1 + returns).cumprod() * initial_capital
    total_return = (equity_curve.iloc[-1] / initial_capital) - 1
    
    n_periods = len(returns)
    if n_periods > 0:
        annual_return = (1 + total_return) ** (252 / n_periods) - 1
        # Calcular retorno mensal médio
        monthly_return = (1 + total_return) ** (21 / n_periods) - 1 if n_periods >= 21 else annual_return / 12
    else:
        annual_return = 0.0
        monthly_return = 0.0
    
    annual_volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0.0
    
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve / running_max - 1)
    max_drawdown = drawdown.min() if len(drawdown) > 0 else 0.0
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'monthly_return': monthly_return,
        'annual_volatility': annual_volatility,
        'max_drawdown': max_drawdown
    }

def get_monthly_returns(returns):
    """Calcula retornos mensais"""
    monthly = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
    return monthly

def load_mt5_data():
    """Carrega dados MT5 do sidebar"""
    uploaded_mt5 = st.sidebar.file_uploader(
        "📊 Upload relatório MT5 (HTML)",        type=['html', 'htm'],
        help="Faça upload do relatório HTML exportado do MetaTrader 5"
    )

    mt5_data = None
    if uploaded_mt5 is not None:
        try:
            file_type = '.html' if uploaded_mt5.name.lower().endswith(('.html', '.htm')) else '.pdf'
            parser = MT5ReportParser(uploaded_mt5, file_type)
            mt5_data = parser.get_portfolio_summary()
            
            # Verificação robusta de que mt5_data é válido
            if mt5_data and isinstance(mt5_data, dict):
                st.sidebar.success("✅ MT5 carregado com sucesso!")
                st.sidebar.write(f"**Conta:** {mt5_data.get('account_name', 'N/A')}")
                st.sidebar.write(f"**Saldo:** R$ {mt5_data.get('balance', 0):,.2f}")
                st.sidebar.write(f"**Lucro:** R$ {mt5_data.get('net_profit', 0):,.2f}")
                st.sidebar.write(f"**Retorno:** {mt5_data.get('gain', '0%')}")
            else:
                st.sidebar.warning("⚠️ Arquivo MT5 processado, mas dados insuficientes extraídos.")
                # Criar dados padrão se mt5_data for None
                mt5_data = {
                    'account_name': 'Conta MT5', 'account_number': 'N/A', 'currency': 'BRL',
                    'balance': 0, 'equity': 0, 'net_profit': 0, 'initial_capital': 0,
                    'gain': '0%', 'drawdown': '0%', 'trading_activity': '0%'
                }
            
        except Exception as e:
            st.sidebar.error(f"❌ Erro ao processar MT5: {str(e)}")
            mt5_data = None
    
    return mt5_data

# Funções auxiliares para reduzir complexidade cognitiva

def _validate_mt5_data():
    """Valida se há dados MT5 carregados"""
    mt5_data = st.session_state.get('mt5_data')
    
    if not mt5_data:
        st.warning("⚠️ Nenhum dado MT5 carregado. Faça upload de um relatório MT5 no sidebar.")
        st.info("📝 **Como usar:** Carregue um relatório MT5 HTML no sidebar para ativar as comparações avançadas.")
        return None
        
    st.success("✅ Dados MT5 carregados com sucesso!")
    return mt5_data

def _validate_pca_data():
    """Valida e carrega dados PCA"""
    RAW_DATA = get_raw_data_path()
    
    if not os.path.exists(RAW_DATA):
        st.error("❌ Dados PCA não encontrados. Vá para a página Home e baixe os dados primeiro.")
        return None, None
    
    df = pd.read_csv(RAW_DATA, index_col=0, parse_dates=True)
    returns = df.pct_change().dropna()
    return df, returns

def _setup_mt5_comparison_sidebar():
    """Configura sidebar para comparação MT5"""
    st.sidebar.subheader("⚙️ Configurações de Comparação")
    
    initial_capital = st.sidebar.number_input(
        '💰 Capital Base (R$)',
        min_value=100.0, 
        max_value=1e7, 
        value=10000.0, 
        step=100.0
    )
    
    show_detailed_metrics = st.sidebar.checkbox("📊 Métricas Detalhadas", value=True)
    
    return initial_capital, show_detailed_metrics

def _select_pca_assets(df):
    """Seleciona ativos para comparação PCA"""
    if 'selected' not in st.session_state:
        st.session_state['selected'] = df.columns.tolist()[:5]
    
    selected = st.sidebar.multiselect(
        'Ativos PCA para comparação', 
        df.columns.tolist(),
        default=st.session_state['selected'][:5]
    )
    
    if len(selected) < 3:
        st.warning('⚠️ Selecione pelo menos 3 ativos para comparação válida')
        return None
    
    return selected

def _calculate_pca_metrics(returns, selected, initial_capital):
    """Calcula métricas do portfolio PCA"""
    returns_selected = returns[selected]
    portf_ret = returns_selected.mean(axis=1)
    portf_cum = (1 + portf_ret).cumprod() * initial_capital
    pca_metrics = calculate_metrics(portf_ret, initial_capital)
    
    return portf_ret, portf_cum, pca_metrics

def _display_pca_summary(pca_metrics, portf_cum, initial_capital):
    """Exibe resumo do portfolio PCA"""
    pca_final_value = portf_cum.iloc[-1]
    pca_return = (pca_final_value / initial_capital) - 1
    
    st.metric(FINAL_CAPITAL_LABEL, format_currency(pca_final_value))
    st.metric(TOTAL_RETURN_LABEL, format_percentage(pca_return))
    st.metric("Retorno Anualizado", format_percentage(pca_metrics['annual_return']))
    st.metric("Retorno Mensal", format_percentage(pca_metrics.get('monthly_return', pca_metrics['annual_return'] / 12)))
    st.metric("Volatilidade", format_percentage(pca_metrics['annual_volatility']))
    st.metric("Max Drawdown", format_percentage(pca_metrics['max_drawdown']))
    
    return pca_return

def _display_mt5_summary(mt5_data):
    """Exibe resumo do MT5"""
    mt5_balance = mt5_data.get('balance', 0)
    mt5_profit = mt5_data.get('net_profit', 0)
    mt5_initial = mt5_data.get('initial_capital', mt5_balance - mt5_profit)
    mt5_return = mt5_profit / mt5_initial if mt5_initial > 0 else 0
    
    st.metric("Saldo Atual", format_currency(mt5_balance))
    st.metric("Lucro Líquido", format_currency(mt5_profit))
    st.metric(TOTAL_RETURN_LABEL, format_percentage(mt5_return))
    
    return mt5_return

def _display_comparative_analysis(pca_metrics, pca_return, mt5_return):
    """Exibe análise comparativa detalhada"""
    st.subheader("🔍 Análise Comparativa Detalhada")
    
    # Performance comparison
    col1, col2 = st.columns(2)
    with col1:
        if pca_return > mt5_return:
            st.success(f"✅ **PCA venceu**: +{(pca_return - mt5_return):.2%} de diferença")
        else:
            st.error(f"❌ **MT5 venceu**: +{(mt5_return - pca_return):.2%} de diferença")
    
    # Risk-adjusted returns
    risk_adjusted_pca = pca_metrics['annual_return'] / pca_metrics['annual_volatility'] if pca_metrics['annual_volatility'] > 0 else 0
    risk_adjusted_mt5 = mt5_return / 0.15  # Assumindo volatilidade de 15% para MT5
    
    with col2:
        if risk_adjusted_pca > risk_adjusted_mt5:
            st.success("✅ **PCA**: Melhor retorno ajustado ao risco")
        else:
            st.info("📊 **MT5**: Melhor retorno ajustado ao risco")

def _display_recommendations(pca_metrics, mt5_return, pca_return):
    """Exibe recomendações baseadas na análise"""
    st.markdown("### 💡 Recomendações")
    recommendations = []
    
    if pca_metrics['annual_volatility'] > 0.3:
        recommendations.append("⚠️ Considere reduzir a volatilidade do portfolio PCA")
    
    if abs(pca_metrics['max_drawdown']) > 0.2:
        recommendations.append("📉 Implemente estratégias de controle de drawdown")
    
    if pca_return < mt5_return:
        recommendations.append("📈 Analise os ativos MT5 para melhorar seleção PCA")
    
    recommendations.append("🔄 Continue monitorando ambas as estratégias")
    
    for rec in recommendations:
        st.write(f"- {rec}")

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
        st.write("🚀 **Análise de Portfólio - by KG**")

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
    
    # Agradecimentos especiais
    st.markdown("---")
    st.subheader("❤️ Agradecimentos Especiais")
    st.markdown("""
    **Agradecimentos sinceros a:**
    
    👨‍💻 **Brenda Souza Barros & Felipe Freitas Alves** pela incrível cumplicidade e apoio contínuo
    no desenvolvimento deste projeto. A parceria de vocês foi fundamental para a criação desta ferramenta.
    """)
    # Download de dados
    st.markdown("---")
    st.subheader("📥 Dados dos Ativos")
    
    if st.button('🔄 Baixar/Atualizar dados dos ativos'):
        with st.spinner('Baixando dados...'):
            from data_fetch import fetch_data
            fetch_data()
        st.success('✅ Dados baixados com sucesso!')    # Status dos dados
    RAW_DATA = get_raw_data_path()
    
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
    st.title(PCA_PERFORMANCE_TITLE)
    
    RAW_DATA = get_raw_data_path()
    
    if not os.path.exists(RAW_DATA):
        st.error(DATA_NOT_FOUND_MSG)
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
        default=st.session_state['selected']    )
    
    if not 3 <= len(selected) <= 20:
        st.warning('⚠️ Selecione entre 3 e 20 ativos para análise PCA')
        return
        
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
    metrics = calculate_metrics(portf_ret, initial_capital)    # Métricas principais
    st.subheader("📊 Métricas de Performance")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(TOTAL_RETURN_LABEL, format_percentage(metrics['total_return']))
    with col2:
        st.metric("Retorno Anualizado", format_percentage(metrics['annual_return']))
    with col3:
        st.metric("Retorno Mensal", format_percentage(metrics['monthly_return']))
    with col4:
        st.metric("Volatilidade", format_percentage(metrics['annual_volatility']))
    with col5:
        st.metric(MAX_DRAWDOWN_LABEL, format_percentage(metrics['max_drawdown']))
    
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
    """Página de comparação avançada com MT5"""
    st.title("⚖️ Comparação Avançada PCA vs MT5")
    
    # Validar dados MT5
    mt5_data = validate_mt5_data()
    if mt5_data is None:
        return
    
    # Validar e carregar dados PCA
    df, returns = validate_pca_data()
    if df is None or returns is None:
        return
    
    # Configurações da sidebar
    initial_capital, show_detailed_metrics = setup_mt5_comparison_sidebar()
    
    # Seleção de ativos para PCA
    selected = select_pca_assets(df)
    if selected is None:
        return
      # Cálculo das métricas do portfolio PCA
    _, portf_cum, pca_metrics = calculate_pca_metrics(returns, selected, initial_capital)
      # === RESUMO COMPARATIVO ===
    st.subheader("📊 Resumo Comparativo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🤖 Portfolio PCA")
        pca_return = display_pca_summary(pca_metrics, portf_cum, initial_capital)
    
    with col2:
        st.markdown("### 📈 MT5 Real")
        mt5_return = display_mt5_summary(mt5_data)
    
    # Gráfico de performance comparativa
    plot_comparative_performance(portf_cum, mt5_data, initial_capital)
    
    # Análise por símbolo MT5
    analyze_symbol_performance(mt5_data)
    
    # Gráfico de radar para métricas de risco
    create_risk_metrics_radar(pca_metrics, mt5_data)
    
    # Análise comparativa detalhada
    if show_detailed_metrics:
        display_comparative_analysis(pca_metrics, pca_return, mt5_return)
    
    # Recomendações
    display_recommendations(pca_metrics, mt5_return, pca_return)

def show_sector_management_page():
    """Página avançada de gestão por setor"""
    from sector_management_helpers import (
        get_risk_profile_config, display_risk_profile, configure_sector_allocation,
        display_allocation_status, create_allocation_visualizations, analyze_sector_performance,
        display_performance_results, generate_recommendations, save_configuration, export_to_excel
    )
    
    st.title("💰 Gestão Avançada por Setor")
    
    RAW_DATA = get_raw_data_path()
    
    if not os.path.exists(RAW_DATA):
        st.error(DATA_NOT_FOUND_MSG)
        return
    
    df = pd.read_csv(RAW_DATA, index_col=0, parse_dates=True)
    returns = df.pct_change().dropna()
    
    st.markdown("""
    💡 **Gestão por Setor**: Aloque seu capital de forma inteligente entre diferentes setores da economia,
    otimizando risco e retorno através de diversificação setorial.
    """)
    
    # Parâmetros principais
    st.sidebar.subheader("⚙️ Configurações")
    total_capital = st.sidebar.number_input(
        '💰 Capital Total (R$)', 
        min_value=1000.0, 
        max_value=1e8, 
        value=100000.0, 
        step=1000.0
    )
    
    rebalance_frequency = st.sidebar.selectbox(
        "🔄 Frequência de Rebalanceamento",
        ["Mensal", "Trimestral", "Semestral", "Anual"]
    )
    
    risk_tolerance = st.sidebar.slider(
        "📊 Tolerância ao Risco",
        min_value=1,
        max_value=10,
        value=5,
        help="1=Conservador, 10=Agressivo"
    )
    
    # === CONFIGURAÇÃO DE ALOCAÇÃO POR SETOR ===
    st.subheader("🏭 Configuração de Alocação por Setor")
    
    # Obter configuração de perfil de risco
    available_sectors = list(ASSET_CATEGORIES.keys())
    risk_config = get_risk_profile_config(risk_tolerance)
    suggested_allocation = risk_config['suggested_allocation']
    risk_profile = risk_config['profile']
    profile_color = risk_config['color']
    
    # Exibir perfil de risco
    display_risk_profile(profile_color, risk_profile)
    
    # Interface para configurar alocação
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("⚖️ Alocação Manual")
        sector_allocations, total_allocation = configure_sector_allocation(
            available_sectors, suggested_allocation
        )
    
    with col2:
        st.subheader("📊 Status da Alocação")
        display_allocation_status(total_allocation, total_capital, sector_allocations)
    
    # === VISUALIZAÇÃO DA ALOCAÇÃO ===
    if total_allocation > 0:
        st.subheader("📈 Visualização da Alocação")
        create_allocation_visualizations(sector_allocations, available_sectors, suggested_allocation)
    
    # === ANÁLISE DE PERFORMANCE POR SETOR ===
    if total_allocation == 100:
        st.markdown("---")
        st.subheader("📊 Análise de Performance por Setor")
        
        allocation_manager, selected_assets, portfolio_weights = analyze_sector_performance(
            df, returns, sector_allocations
        )
        
        if allocation_manager and selected_assets and portfolio_weights:
            display_performance_results(allocation_manager, selected_assets, portfolio_weights)
    
    # === RECOMENDAÇÕES E INSIGHTS ===
    st.markdown("---")
    st.subheader("💡 Recomendações Personalizadas")
    
    recommendations = generate_recommendations(risk_tolerance, sector_allocations, rebalance_frequency)
    for rec in recommendations:
        st.write(f"• {rec}")
    
    # === EXPORT E SALVAMENTO ===
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 Salvar Configuração"):
            save_configuration(sector_allocations, total_capital, risk_tolerance, rebalance_frequency)
    
    with col2:
        if st.button("📥 Exportar para Excel"):
            export_to_excel(total_allocation, sector_allocations, total_capital, risk_profile, rebalance_frequency)

def show_advanced_pca_page():
    """Página de PCA avançado"""    
    from advanced_pca_simplificado import (
        setup_pca_sidebar, execute_static_pca_analysis, display_pca_loadings,
        execute_rolling_pca_analysis, display_rolling_pca_stability, build_pca_portfolio,
        display_portfolio_results, analyze_pca_risk, display_interactive_pca_example
    )
    
    st.title(ADVANCED_PCA_TITLE)
    
    RAW_DATA = get_raw_data_path()
    
    if not os.path.exists(RAW_DATA):
        st.error(DATA_NOT_FOUND_MSG)
        return
    
    try:
        df = pd.read_csv(RAW_DATA, index_col=0, parse_dates=True)
        returns = df.pct_change().dropna()
        
        # Header com informações didáticas
        st.markdown("""
        ### 🧠 Análise PCA Avançada
        
        **Análise de Componentes Principais (PCA)** é uma técnica estatística que reduz a dimensionalidade dos dados 
        identificando as direções de maior variância. Na análise de portfolios:
        
        - **Componentes principais**: Combinações lineares dos ativos originais
        - **Variância explicada**: Quanto cada componente captura da variação total
        - **Loadings**: Peso de cada ativo em cada componente
        - **PCA rolling**: Evolução temporal dos componentes
        """)
        
        # Tabs para organizar análises
        pca_tabs = st.tabs([
            "📊 Análise Estática", 
            "📈 PCA Rolling", 
            "🎯 Seleção de Portfolio", 
            "📉 Análise de Risco",
            "🔬 Explicação Didática"
        ])
        
        # Configuração no sidebar
        sidebar_config = setup_pca_sidebar(df)
        if sidebar_config[0] is None:
            st.warning("⚠️ Selecione pelo menos 3 ativos para análise PCA")
            return
        
        selected_assets, n_components, rebalance_freq, window_size, rebalance_window = sidebar_config
        returns_selected = returns[selected_assets]
        
        # Tab 1: Análise Estática
        with pca_tabs[0]:
            st.subheader("📊 Análise PCA Estática")
            pca, components, explained_var = execute_static_pca_analysis(
                returns_selected, selected_assets, n_components
            )
            display_pca_loadings(pca, selected_assets, explained_var)
        
        # Tab 2: PCA Rolling
        with pca_tabs[1]:
            rolling_variance, rolling_loadings_pc1, rolling_dates = execute_rolling_pca_analysis(
                returns_selected, selected_assets, window_size, rebalance_freq, rebalance_window
            )
            display_rolling_pca_stability(rolling_variance, rolling_loadings_pc1, rolling_dates, selected_assets)
        
        # Tab 3: Seleção de Portfolio
        with pca_tabs[2]:
            strategy_type = st.selectbox(
                "Estratégia de construção:",
                ["Maximum Diversification", "Minimum Variance", "Equal Risk Contribution"]
            )
            
            portfolio_results = build_pca_portfolio(
                pca, components, returns_selected, selected_assets, strategy_type, rebalance_freq
            )
            
            if portfolio_results[0] is not None:
                weights, portfolio_cumulative, total_return, annual_return, annual_vol, sharpe = portfolio_results
                display_portfolio_results(
                    weights, portfolio_cumulative, total_return, annual_return, 
                    annual_vol, sharpe, selected_assets, strategy_type
                )
        
        # Tab 4: Análise de Risco
        with pca_tabs[3]:
            analyze_pca_risk(pca, selected_assets, n_components)
        
        # Tab 5: Explicação Didática
        with pca_tabs[4]:
            display_interactive_pca_example(pca, selected_assets, n_components)
                
    except Exception as e:
        st.error(f"❌ Erro ao carregar dados: {str(e)}")

def show_pair_trading_page():
    """Página de pair trading"""
    st.title(PAIR_TRADING_TITLE)
    
    RAW_DATA = get_raw_data_path()
    
    if not os.path.exists(RAW_DATA):
        st.error(DATA_NOT_FOUND_MSG)
        return
    
    try:
        df = pd.read_csv(RAW_DATA, index_col=0, parse_dates=True)
        
        # Header com informações didáticas
        st.markdown("""
        ### 👫 Estratégia de Pair Trading
        
        **Pair Trading** é uma estratégia market-neutral que busca lucrar com a convergência de preços entre ativos correlacionados:
        
        - **Cointegração**: Relação de longo prazo entre ativos
        - **Spread**: Diferença de preços normalizada entre os ativos
        - **Z-Score**: Desvio do spread em relação à média histórica
        - **Mean Reversion**: Tendência do spread retornar à média
        """)
        
        # Tabs principais
        pair_tabs = st.tabs([
            "🔍 Identificar Pares", 
            "📊 Análise Detalhada", 
            "⚡ Sinais de Trading", 
            "📈 Backtest", 
            "🎯 Otimização",
            "📚 Tutorial"
        ])
        
        # Configurações na sidebar usando helper
        from pair_trading_helpers import setup_pair_trading_sidebar
        params = setup_pair_trading_sidebar()
        # Inicializar análise
        pair_analyzer = PairTradingAnalysis(df)
        all_assets = df.columns.tolist()
        
        # Garantir que a variável all_assets esteja disponível em todas as tabs
        st.session_state['all_assets'] = all_assets
        
        # Importar funções auxiliares do módulo pair_trading_helpers
        from pair_trading_helpers import (
            find_cointegrated_pairs_tab, 
            detailed_analysis_tab, 
            trading_signals_tab, 
            backtest_tab, 
            optimization_tab, 
            tutorial_tab
        )
        
        # Tab 1: Identificar Pares
        with pair_tabs[0]:
            find_cointegrated_pairs_tab(pair_analyzer, all_assets, params)
          # Tab 2: Análise Detalhada
        with pair_tabs[1]:
            asset1, asset2 = detailed_analysis_tab(pair_analyzer, all_assets, params)
        
        # Tab 3: Sinais de Trading
        with pair_tabs[2]:
            # Verificar se asset1 e asset2 estão definidos
            if 'asset1' not in locals() or 'asset2' not in locals() or asset1 is None or asset2 is None:
                if len(all_assets) >= 2:
                    asset1 = all_assets[0]
                    asset2 = all_assets[1]
                else:
                    st.error("❌ Não foi possível selecionar dois ativos. Selecione ativos manualmente na aba 'Análise Detalhada'.")
                    asset1, asset2 = None, None
            
            if asset1 and asset2:
                trading_signals_tab(pair_analyzer, asset1, asset2, params)
          # Tab 4: Backtest
        with pair_tabs[3]:
            if asset1 and asset2:
                backtest_tab(pair_analyzer, asset1, asset2, params, pair_analyzer.price_data)
        
        # Tab 5: Otimização
        with pair_tabs[4]:
            if asset1 and asset2:
                optimization_tab(pair_analyzer, asset1, asset2, params)
          # Tab 6: Tutorial
        with pair_tabs[5]:
            tutorial_tab(all_assets)
            
    except Exception as e:
        st.error(f"❌ Erro ao carregar dados: {str(e)}")

def show_statistical_analysis_page():
    """Página de análise estatística avançada"""
    st.title(STATISTICAL_ANALYSIS_TITLE)
    
    RAW_DATA = get_raw_data_path()
    
    if not os.path.exists(RAW_DATA):
        st.error(DATA_NOT_FOUND_MSG)
        return
    
    try:
        df = pd.read_csv(RAW_DATA, index_col=0, parse_dates=True)
        
        # Header informativo usando helper
        from statistical_analysis_helpers import display_statistical_header
        display_statistical_header()
        
        # Inicializar análise estatística
        stat_analyzer = StatisticalAnalysis(df)
          # Tabs principais
        stat_tabs = st.tabs([
            "🎯 Análise de Extremos", 
            "📈 Comparação de Distribuições", 
            "🔬 Modelos de Risco",
            "🔄 Pair Trading Avançado",
            "📚 Documentação"
        ])
        
        # Tab 1: Análise de Extremos usando helper
        from statistical_analysis_helpers import extreme_analysis_tab
        with stat_tabs[0]:
            extreme_analysis_tab(stat_analyzer, df)
        
        # Tab 2: Comparação de Distribuições usando helper
        from statistical_analysis_helpers import distribution_comparison_tab
        with stat_tabs[1]:
            distribution_comparison_tab(stat_analyzer)
        
        # Tab 3: Modelos de Risco usando helper
        from statistical_analysis_helpers import risk_models_tab
        with stat_tabs[2]:
            risk_models_tab(stat_analyzer, df)
        
        # Tab 4: Pair Trading Avançado usando helper
        from statistical_analysis_helpers import advanced_pair_trading_tab
        with stat_tabs[3]:
            advanced_pair_trading_tab(df)          # Tab 5: Documentação usando helper
        from statistical_analysis_helpers import documentation_tab
        with stat_tabs[4]:
            documentation_tab()
          # Adicionar botão para download da análise estatística em PDF
        st.sidebar.markdown("### 📥 Download da Análise")
        st.sidebar.markdown("Baixe a análise estatística completa em formato PDF")
        
        # Determinar qual ativo está selecionado atualmente - usando Session State
        if 'selected_asset_for_pdf' not in st.session_state:
            # Tentar obter um ativo disponível da lista de colunas do dataframe
            if df is not None and not df.empty:
                assets_list = df.columns.tolist()
                if PETR4_SYMBOL in assets_list:
                    st.session_state['selected_asset_for_pdf'] = PETR4_SYMBOL
                elif len(assets_list) > 0:
                    st.session_state['selected_asset_for_pdf'] = assets_list[0]
                else:
                    st.session_state['selected_asset_for_pdf'] = "ATIVO"
            else:
                st.session_state['selected_asset_for_pdf'] = "ATIVO"
        
        # Permitir ao usuário selecionar um ativo específico para o PDF
        available_assets = [col for col in df.columns if df[col].dtype in ['float64', 'int64']] if not df.empty else []
        st.sidebar.selectbox("Ativo para o PDF:", available_assets, 
                            key="selected_asset_for_pdf",
                            index=0 if not available_assets or PETR4_SYMBOL not in available_assets else available_assets.index(PETR4_SYMBOL))
        
        if st.sidebar.button("📄 Download PDF da Análise Estatística", type="primary"):
            # Código para gerar o PDF completo com dados reais e visualizações
            with st.spinner("Gerando PDF da análise estatística com gráficos e dados..."):
                try:
                    import base64
                    from pdf_export_helpers import generate_complete_statistical_analysis_pdf
                    
                    # Usar o ativo selecionado do session_state
                    selected_asset = st.session_state.get('selected_asset_for_pdf')
                      # Gerar PDF com dados reais e visualizações
                    pdf_data = generate_complete_statistical_analysis_pdf(df, selected_asset)                    # Criar botão de download usando a função auxiliar
                    # Garantir que selected_asset é string e não é None antes de usar replace()
                    if selected_asset is None:
                        file_suffix = 'geral'
                    else:
                        # Converter para string e substituir pontos por underscores
                        try:
                            file_suffix = str(selected_asset).replace('.', '_')
                        except:
                            file_suffix = 'geral'
                            
                    download_button_html = create_download_button(
                        pdf_data,
                        filename=f"analise_estatistica_{file_suffix}.pdf", 
                        button_text="Baixar Análise Estatística Completa (PDF)"
                    )
                    
                    st.sidebar.markdown(download_button_html, unsafe_allow_html=True)
                    st.sidebar.success("PDF gerado com sucesso! Clique no botão acima para baixar.")
                except Exception as pdf_error:
                    st.sidebar.error(f"Erro ao gerar PDF: {pdf_error}")
    
    except Exception as e:
        st.error(f"❌ Erro ao carregar dados: {str(e)}")

# =====================================================================
# FUNÇÃO PRINCIPAL E NAVEGAÇÃO
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
            "🔄 Pair Trading",
            "📈 Análise Estatística",
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
    elif page == "📈 Análise Estatística":
        show_statistical_analysis_page()
    # elif page == "📋 Relatório Trading":
    #     show_trading_report_page()

# =====================================================================
# EXECUÇÃO DA APLICAÇÃO
# =====================================================================

if __name__ == "__main__":
    main()
