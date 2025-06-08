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
    initial_sidebar_state="auto"
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
        'max_drawdown': max_drawdown
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
    """Página de comparação avançada com MT5"""
    st.title("⚖️ Comparação Avançada PCA vs MT5")
    
    # Verificar se há dados MT5
    mt5_data = st.session_state.get('mt5_data')
    if not mt5_data:
        st.warning("⚠️ Nenhum dado MT5 carregado. Faça upload de um relatório MT5 no sidebar.")
        st.info("📝 **Como usar:** Carregue um relatório MT5 HTML no sidebar para ativar as comparações avançadas.")
        return
    
    st.success("✅ Dados MT5 carregados com sucesso!")
    
    # Verificar dados PCA
    DATA_DIR = 'data'
    RAW_DATA = os.path.join(DATA_DIR, 'raw_data.csv')
    
    if not os.path.exists(RAW_DATA):
        st.error("❌ Dados PCA não encontrados. Vá para a página Home e baixe os dados primeiro.")
        return
    
    # Carregar dados PCA
    df = pd.read_csv(RAW_DATA, index_col=0, parse_dates=True)
    returns = df.pct_change().dropna()
    
    # Configurações básicas
    st.sidebar.subheader("⚙️ Configurações de Comparação")
    initial_capital = st.sidebar.number_input(
        '💰 Capital Base (R$)', 
        min_value=100.0, 
        max_value=1e7, 
        value=10000.0, 
        step=100.0
    )
    
    normalize_comparison = st.sidebar.checkbox("🔄 Normalizar Comparação", value=True)
    show_detailed_metrics = st.sidebar.checkbox("📊 Métricas Detalhadas", value=True)
    
    # Seleção de ativos para PCA
    if 'selected' not in st.session_state:
        st.session_state['selected'] = df.columns.tolist()[:5]
    
    selected = st.sidebar.multiselect(
        'Ativos PCA para comparação', 
        df.columns.tolist(),
        default=st.session_state['selected'][:5]
    )
    
    if len(selected) < 3:
        st.warning('⚠️ Selecione pelo menos 3 ativos para comparação válida')
        return
    
    # Calcular métricas PCA
    df_selected = df[selected]
    returns_selected = returns[selected]
    portf_ret = returns_selected.mean(axis=1)
    portf_cum = (1 + portf_ret).cumprod() * initial_capital
    pca_metrics = calculate_metrics(portf_ret, initial_capital)
    
    # === RESUMO COMPARATIVO ===
    st.subheader("📊 Resumo Comparativo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🤖 Portfolio PCA")
        pca_final_value = portf_cum.iloc[-1]
        pca_return = (pca_final_value / initial_capital) - 1
        
        st.metric("Capital Final", f"R$ {pca_final_value:,.2f}")
        st.metric("Retorno Total", f"{pca_return:.2%}")
        st.metric("Retorno Anualizado", f"{pca_metrics['annual_return']:.2%}")
        st.metric("Volatilidade", f"{pca_metrics['annual_volatility']:.2%}")
        st.metric("Max Drawdown", f"{pca_metrics['max_drawdown']:.2%}")
    
    with col2:
        st.markdown("### 📈 MT5 Real")
        mt5_balance = mt5_data.get('balance', 0)
        mt5_profit = mt5_data.get('net_profit', 0)
        mt5_initial = mt5_data.get('initial_capital', mt5_balance - mt5_profit)
        mt5_return = mt5_profit / mt5_initial if mt5_initial > 0 else 0
        
        st.metric("Saldo Atual", f"R$ {mt5_balance:,.2f}")
        st.metric("Lucro Líquido", f"R$ {mt5_profit:,.2f}")
        st.metric("Retorno Total", f"{mt5_return:.2%}")
        st.metric("Ganho", mt5_data.get('gain', 'N/A'))
        st.metric("Drawdown", mt5_data.get('drawdown', 'N/A'))
    
    # === GRÁFICOS DE COMPARAÇÃO ===
    st.markdown("---")
    st.subheader("📈 Análise Temporal Comparativa")
    
    # Gráfico de evolução temporal
    temporal_fig = plot_temporal_comparison(portf_ret, mt5_data, initial_capital)
    if temporal_fig:
        st.plotly_chart(temporal_fig, use_container_width=True, key="temporal_comparison")
        
        st.info("""
        💡 **Interpretação:**
        - **Linha Azul (PCA)**: Evolução baseada na média dos retornos dos ativos selecionados
        - **Linha Vermelha (MT5)**: Curva simulada baseada nos resultados reais do MT5
        - **Divergências**: Indicam diferentes estratégias e timing de entrada/saída
        """)
    
    # Comparação de Drawdown
    st.subheader("📉 Comparação de Drawdown")
    drawdown_fig = plot_drawdown_comparison(portf_ret, mt5_data)
    if drawdown_fig:
        st.plotly_chart(drawdown_fig, use_container_width=True, key="drawdown_comparison")
        
        st.info("""
        💡 **Análise de Drawdown:**
        - **Drawdown**: Perda máxima do pico até o vale
        - **PCA**: Baseado nos dados históricos reais
        - **MT5**: Simulação baseada no drawdown máximo reportado
        """)
    
    # === ANÁLISE COMPARATIVA DETALHADA ===
    if show_detailed_metrics:
        st.markdown("---")
        st.subheader("🔬 Análise Comparativa Detalhada")
        
        tab1, tab2, tab3 = st.tabs(["📊 Métricas de Risco", "🎯 Análise Radar", "📋 Alocação MT5"])
        
        with tab1:
            st.markdown("### 📊 Métricas de Risco Detalhadas")
            
            risk_metrics = create_risk_metrics_analysis(pca_metrics, mt5_data)
            if risk_metrics:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**PCA Portfolio**")
                    st.write(f"• Sharpe Ratio: {risk_metrics['pca']['sharpe_ratio']:.2f}")
                    st.write(f"• Recovery Factor: {risk_metrics['pca']['recovery_factor']:.2f}")
                    st.write(f"• Calmar Ratio: {risk_metrics['pca']['calmar_ratio']:.2f}")
                    st.write(f"• Volatilidade: {risk_metrics['pca']['volatility']:.2%}")
                    
                with col2:
                    st.markdown("**MT5 Real**")
                    st.write(f"• Profit Factor: {risk_metrics['mt5']['profit_factor']:.2f}")
                    st.write(f"• Recovery Factor: {risk_metrics['mt5']['recovery_factor']:.2f}")
                    st.write(f"• Win Rate: {risk_metrics['mt5']['win_rate']:.2%}")
                    st.write(f"• Max Drawdown: {risk_metrics['mt5']['drawdown']:.2%}")
                    
                # Explicações didáticas
                st.markdown("---")
                st.info("""
                📚 **Explicação das Métricas:**
                - **Sharpe Ratio**: Retorno ajustado pelo risco (>1 é bom, >2 é excelente)
                - **Recovery Factor**: Capacidade de recuperação de perdas
                - **Calmar Ratio**: Retorno anualizado / Max Drawdown
                - **Profit Factor**: Total de ganhos / Total de perdas (>1.5 é bom)
                - **Win Rate**: Percentual de trades vencedores
                """)
        
        with tab2:
            st.markdown("### 🎯 Análise Multidimensional")
            
            risk_metrics = create_risk_metrics_analysis(pca_metrics, mt5_data)
            if risk_metrics:
                radar_fig = create_performance_radar_chart(pca_metrics, mt5_data, risk_metrics)
                if radar_fig:
                    st.plotly_chart(radar_fig, use_container_width=True, key="radar_chart")
                    
                    st.info("""
                    **Interpretação do Gráfico Radar:**
                    - **Retorno**: Performance de retorno normalizada
                    - **Risco (inv)**: Inverso da volatilidade (maior valor = menor risco)
                    - **Sharpe**: Índice de Sharpe ou Profit Factor
                    - **Recuperação**: Capacidade de recuperação de drawdowns
                    - **Consistência**: Estabilidade dos resultados
                    """)
        
        with tab3:
            st.markdown("### 📊 Análise de Alocação MT5")
            
            allocation_analysis = create_portfolio_allocation_analysis(mt5_data)
            if allocation_analysis:
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(allocation_analysis['pie'], use_container_width=True, key="allocation_pie")
                with col2:
                    st.plotly_chart(allocation_analysis['bar'], use_container_width=True, key="allocation_bar")
                
                st.markdown("### 📋 Detalhamento por Símbolo")
                st.dataframe(allocation_analysis['data'], use_container_width=True)
                
                # Insights automáticos
                best_symbol = allocation_analysis['data'].loc[allocation_analysis['data']['P&L'].idxmax(), SYMBOL_COLUMN]
                worst_symbol = allocation_analysis['data'].loc[allocation_analysis['data']['P&L'].idxmin(), SYMBOL_COLUMN]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"🎯 **Melhor Símbolo**: {best_symbol}")
                with col2:
                    st.error(f"⚠️ **Pior Símbolo**: {worst_symbol}")
            else:
                st.warning("⚠️ Dados de alocação não disponíveis no relatório MT5")
    
    # === CONCLUSÕES E RECOMENDAÇÕES ===
    st.markdown("---")
    st.subheader("🎯 Conclusões e Recomendações")
    
    # Comparar performance
    pca_performance = pca_return
    mt5_performance = mt5_return
    
    col1, col2 = st.columns(2)
    with col1:
        if pca_performance > mt5_performance:
            st.success(f"✅ **PCA Supera MT5**: {(pca_performance - mt5_performance)*100:.2f}% a mais")
        else:
            st.info(f"📊 **MT5 Supera PCA**: {(mt5_performance - pca_performance)*100:.2f}% a mais")
    
    with col2:
        risk_adjusted_pca = pca_performance / max(abs(pca_metrics['max_drawdown']), 0.01)
        risk_adjusted_mt5 = mt5_performance / max(float(mt5_data.get('drawdown', '1%').replace('%', ''))/100, 0.01)
        
        if risk_adjusted_pca > risk_adjusted_mt5:
            st.success("✅ **PCA**: Melhor retorno ajustado ao risco")
        else:
            st.info("📊 **MT5**: Melhor retorno ajustado ao risco")
    
    # Recomendações
    st.markdown("### 💡 Recomendações")
    recommendations = []
    
    if pca_metrics['annual_volatility'] > 0.3:
        recommendations.append("⚠️ Considere reduzir a volatilidade do portfolio PCA")
    
    if abs(pca_metrics['max_drawdown']) > 0.2:
        recommendations.append("📉 Implemente estratégias de controle de drawdown")
    
    if pca_performance < mt5_performance:
        recommendations.append("📈 Analise os ativos MT5 para melhorar seleção PCA")
    
    recommendations.append("🔄 Continue monitorando ambas as estratégias")
    recommendations.append("📊 Considere combinar insights de ambas as abordagens")
    
    for rec in recommendations:
        st.write(f"• {rec}")
    
    # Export de dados
    st.markdown("---")
    if st.button("📥 Exportar Comparação"):
        comparison_data = {
            'PCA_Return': pca_performance,
            'MT5_Return': mt5_performance,
            'PCA_Volatility': pca_metrics['annual_volatility'],
            'PCA_Drawdown': pca_metrics['max_drawdown'],
            'MT5_Drawdown': mt5_data.get('drawdown', 'N/A'),
            'Assets_Selected': selected
        }
        
        # Salvar em JSON para posterior análise
        import json
        with open('comparison_results.json', 'w') as f:
            json.dump(comparison_data, f, indent=2, default=str)
        
        st.success("✅ Dados de comparação exportados para 'comparison_results.json'")

def show_sector_management_page():
    """Página avançada de gestão por setor"""
    st.title("💰 Gestão Avançada por Setor")
    
    DATA_DIR = 'data'
    RAW_DATA = os.path.join(DATA_DIR, 'raw_data.csv')
    
    if not os.path.exists(RAW_DATA):
        st.error("❌ Dados não encontrados. Vá para a página Home e baixe os dados primeiro.")
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
    
    # Setores disponíveis com sugestões baseadas no perfil de risco
    available_sectors = list(ASSET_CATEGORIES.keys())
    
    # Sugestões de alocação baseadas no perfil de risco
    if risk_tolerance <= 3:  # Conservador
        suggested_allocation = {
            'financeiro': 30, 'energia': 20, 'consumo': 15, 
            'materiais': 10, 'telecomunicações': 10, 'tecnologia': 15
        }
        risk_profile = "Conservador"
        profile_color = "green"
    elif risk_tolerance <= 7:  # Moderado
        suggested_allocation = {
            'financeiro': 25, 'energia': 15, 'consumo': 20, 
            'materiais': 15, 'telecomunicações': 10, 'tecnologia': 15
        }
        risk_profile = "Moderado"
        profile_color = "orange"
    else:  # Agressivo
        suggested_allocation = {
            'financeiro': 20, 'energia': 10, 'consumo': 15, 
            'materiais': 20, 'telecomunicações': 15, 'tecnologia': 20
        }
        risk_profile = "Agressivo"
        profile_color = "red"
    
    st.info(f"📊 **Perfil de Risco**: {risk_profile} | **Sugestão**: Baseada em sua tolerância ao risco")
    
    # Interface para configurar alocação
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("⚖️ Alocação Manual")
        sector_allocations = {}
        total_allocation = 0
        
        # Usar sugestão como base ou permitir configuração manual
        use_suggestion = st.checkbox("🎯 Usar Alocação Sugerida", value=True)
        
        for i, sector in enumerate(available_sectors):
            if use_suggestion and sector in suggested_allocation:
                default_value = suggested_allocation[sector]
            else:
                default_value = 100 // len(available_sectors)
                
            allocation = st.slider(
                f"{sector.title()} (%)",
                min_value=0,
                max_value=100,
                value=default_value,
                key=f"allocation_{sector}"
            )
            sector_allocations[sector] = allocation
            total_allocation += allocation
    
    with col2:
        st.subheader("📊 Status da Alocação")
        
        if total_allocation == 100:
            st.success("✅ Alocação Balanceada!")
        elif total_allocation < 100:
            st.warning(f"⚠️ Faltam {100-total_allocation}%")
        else:
            st.error(f"❌ Excesso de {total_allocation-100}%")
        
        st.metric("Total Alocado", f"{total_allocation}%")
        
        # Mostrar valor por setor
        st.markdown("**Valores por Setor:**")
        for sector, pct in sector_allocations.items():
            value = total_capital * (pct / 100)
            if pct > 0:
                st.write(f"• {sector.title()}: R$ {value:,.0f}")
    
    # === VISUALIZAÇÃO DA ALOCAÇÃO ===
    if total_allocation > 0:
        st.subheader("📈 Visualização da Alocação")
        
        tab1, tab2, tab3 = st.tabs(["🥧 Pizza", "📊 Barras", "🎯 Comparação"])
        
        with tab1:
            # Filtrar setores com alocação > 0
            active_sectors = {k: v for k, v in sector_allocations.items() if v > 0}
            
            if active_sectors:
                fig_pie = px.pie(
                    values=list(active_sectors.values()),
                    names=[s.title() for s in active_sectors.keys()],
                    title='Distribuição por Setor',
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig_pie, use_container_width=True)
        
        with tab2:
            if active_sectors:
                fig_bar = px.bar(
                    x=list(active_sectors.values()),
                    y=[s.title() for s in active_sectors.keys()],
                    title='Alocação por Setor (%)',
                    orientation='h',
                    color=list(active_sectors.values()),
                    color_continuous_scale="viridis"
                )
                st.plotly_chart(fig_bar, use_container_width=True)
        
        with tab3:
            # Comparar com alocação sugerida
            comparison_df = pd.DataFrame({
                'Setor': [s.title() for s in available_sectors],
                'Sua Alocação (%)': [sector_allocations.get(s, 0) for s in available_sectors],
                'Sugerida (%)': [suggested_allocation.get(s, 0) for s in available_sectors]
            })
            
            fig_comparison = px.bar(
                comparison_df,
                x='Setor',
                y=['Sua Alocação (%)', 'Sugerida (%)'],
                title='Comparação: Sua Alocação vs Sugerida',
                barmode='group'
            )
            st.plotly_chart(fig_comparison, use_container_width=True)
    
    # === ANÁLISE DE PERFORMANCE POR SETOR ===
    if total_allocation == 100:
        st.markdown("---")
        st.subheader("📊 Análise de Performance por Setor")
        
        try:
            # Inicializar gestor de alocação
            allocation_manager = PortfolioAllocationManager(df, returns)
            
            # Configurar alocação por setor
            sector_budgets = {sector: pct/100 for sector, pct in sector_allocations.items() if pct > 0}
            allocation_manager.set_sector_allocation(sector_budgets)
            
            # Selecionar ativos por setor
            selected_assets = []
            for sector in sector_budgets.keys():
                if sector in ASSET_CATEGORIES:
                    sector_assets = [asset for asset in ASSET_CATEGORIES[sector] if asset in df.columns]
                    selected_assets.extend(sector_assets[:3])  # Top 3 por setor
            
            if len(selected_assets) >= 3:
                # Calcular pesos do portfolio
                allocation_method = st.selectbox(
                    "Método de Alocação:",
                    ["equal_weight", "market_cap", "risk_parity"]
                )
                
                portfolio_weights = allocation_manager.calculate_portfolio_weights(
                    selected_assets, allocation_method
                )
                
                # Mostrar alocação
                st.subheader("📊 Visualização da Alocação Calculada")
                
                # Gráfico de pizza da alocação
                fig_allocation = allocation_manager.plot_sector_allocation()
                if fig_allocation:
                    st.plotly_chart(fig_allocation, use_container_width=True, key="sector_allocation_chart")
                
                # Calcular performance por setor
                sector_performance = allocation_manager.calculate_sector_performance(
                    selected_assets, portfolio_weights
                )
                
                if sector_performance:
                    # Comparação de performance
                    fig_comparison = allocation_manager.plot_sector_performance_comparison(sector_performance)
                    if fig_comparison:
                        st.plotly_chart(fig_comparison, use_container_width=True, key="sector_performance_comparison")
                    
                    # Tabela de performance
                    st.subheader("📋 Performance Detalhada por Setor")
                    performance_df = pd.DataFrame(sector_performance).T
                    performance_df = performance_df.round(4)
                    st.dataframe(performance_df, use_container_width=True)
                    
                    # Insights automáticos
                    best_sector = performance_df['annual_return'].idxmax()
                    worst_sector = performance_df['annual_return'].idxmin()
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.success(f"🎯 **Melhor Setor**: {best_sector}")
                        st.write(f"Retorno: {performance_df.loc[best_sector, 'annual_return']:.2%}")
                    with col2:
                        st.warning(f"⚠️ **Setor Desafiador**: {worst_sector}")
                        st.write(f"Retorno: {performance_df.loc[worst_sector, 'annual_return']:.2%}")
            
            else:
                st.warning("⚠️ Não há ativos suficientes nos setores selecionados para análise")
                
        except Exception as e:
            st.error(f"❌ Erro na análise de alocação: {str(e)}")
    
    # === RECOMENDAÇÕES E INSIGHTS ===
    st.markdown("---")
    st.subheader("💡 Recomendações Personalizadas")
    
    recommendations = []
    
    # Baseado no perfil de risco
    if risk_tolerance <= 3:
        recommendations.append("🛡️ **Conservador**: Considere aumentar alocação em setores defensivos (energia, financeiro)")
        recommendations.append("📊 Mantenha diversificação para reduzir volatilidade")
    elif risk_tolerance <= 7:
        recommendations.append("⚖️ **Moderado**: Balance entre crescimento e estabilidade")
        recommendations.append("🔄 Rebalanceie trimestralmente para manter alocação alvo")
    else:
        recommendations.append("🚀 **Agressivo**: Foque em setores de crescimento (tecnologia, materiais)")
        recommendations.append("📈 Aceite maior volatilidade para potencial maior retorno")
    
    # Baseado na alocação atual
    max_allocation = max(sector_allocations.values()) if sector_allocations.values() else 0
    if max_allocation > 40:
        recommendations.append("⚠️ **Concentração Alta**: Considere diversificar mais entre setores")
    
    if len([v for v in sector_allocations.values() if v > 0]) < 3:
        recommendations.append("🎯 **Diversificação**: Inclua pelo menos 3-4 setores diferentes")
    
    recommendations.append(f"📅 **Rebalanceamento**: Revise sua alocação {rebalance_frequency.lower()}")
    recommendations.append("📊 **Monitoramento**: Acompanhe performance relativa entre setores")
    
    for rec in recommendations:
        st.write(f"• {rec}")
    
    # === EXPORT E SALVAMENTO ===
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("💾 Salvar Configuração"):
            config = {
                'sector_allocations': sector_allocations,
                'total_capital': total_capital,
                'risk_tolerance': risk_tolerance,
                'rebalance_frequency': rebalance_frequency,
                'timestamp': datetime.now().isoformat()
            }
            
            import json
            with open('sector_allocation_config.json', 'w') as f:
                json.dump(config, f, indent=2)
            
            st.success("✅ Configuração salva!")
    
    with col2:
        if st.button("📥 Exportar para Excel"):
            if total_allocation == 100:
                allocation_data = pd.DataFrame([
                    {
                        'Setor': sector.title(),
                        'Alocação (%)': pct,
                        'Valor (R$)': total_capital * (pct / 100),
                        'Perfil_Risco': risk_profile
                    }
                    for sector, pct in sector_allocations.items() if pct > 0
                ])
                
                allocation_data.to_excel('alocacao_setorial.xlsx', index=False)
                st.success("✅ Dados exportados para 'alocacao_setorial.xlsx'")
            else:
                st.warning("⚠️ Complete a alocação (100%) antes de exportar")

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
        
        # Seleção de ativos
        with st.sidebar:
            st.subheader("⚙️ Configurações PCA")
            selected_assets = st.multiselect(
                "Selecione ativos (mín. 5):",
                df.columns.tolist(),
                default=df.columns.tolist()[:8] if len(df.columns) >= 8 else df.columns.tolist()
            )
            
            n_components = st.slider(
                "Número de componentes:", 
                2, min(10, len(selected_assets)), 
                min(5, len(selected_assets))
            ) if len(selected_assets) >= 2 else 2
            
            rebalance_freq = st.selectbox(
                "Frequência de rebalanceamento:",
                ["Mensal", "Trimestral", "Semestral"],
                index=1
            )
            
            window_size = st.slider("Janela rolling (dias):", 30, 252, 90)
        
        if len(selected_assets) < 5:
            st.warning("⚠️ Selecione pelo menos 5 ativos para análise PCA robusta")
            return
            
        df_selected = df[selected_assets]
        returns_selected = returns[selected_assets].fillna(0)
        
        # Tab 1: Análise Estática
        with pca_tabs[0]:
            st.subheader("📊 Análise PCA Estática")
            
            # Executar PCA
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(returns_selected)
            pca = PCA(n_components=n_components, random_state=42)
            components = pca.fit_transform(scaled_data)
            
            # Métricas principais
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Variância Explicada", f"{pca.explained_variance_ratio_.sum():.1%}")
            with col2:
                st.metric("Primeiro Componente", f"{pca.explained_variance_ratio_[0]:.1%}")
            with col3:
                st.metric("Segundo Componente", f"{pca.explained_variance_ratio_[1]:.1%}")
            with col4:
                st.metric("Número de Ativos", len(selected_assets))
            
            # Gráfico Scree Plot
            fig_scree = px.bar(
                x=list(range(1, len(pca.explained_variance_ratio_) + 1)),
                y=pca.explained_variance_ratio_ * 100,
                title="📈 Scree Plot - Variância Explicada por Componente",
                labels={'x': 'Componente Principal', 'y': 'Variância Explicada (%)'}
            )
            fig_scree.update_layout(height=400)
            st.plotly_chart(fig_scree, use_container_width=True)
            
            # Biplot (loadings)
            col1, col2 = st.columns(2)
            
            with col1:
                # Loadings do primeiro componente
                loadings_pc1 = pd.DataFrame({
                    'Ativo': selected_assets,
                    'Loading PC1': pca.components_[0]
                }).sort_values('Loading PC1', key=abs, ascending=False)
                
                fig_loadings1 = px.bar(
                    loadings_pc1, x='Loading PC1', y='Ativo',
                    title="🎯 Loadings - Primeiro Componente",
                    orientation='h',
                    color='Loading PC1',
                    color_continuous_scale='RdBu_r'
                )
                st.plotly_chart(fig_loadings1, use_container_width=True)
            
            with col2:
                # Loadings do segundo componente
                loadings_pc2 = pd.DataFrame({
                    'Ativo': selected_assets,
                    'Loading PC2': pca.components_[1]
                }).sort_values('Loading PC2', key=abs, ascending=False)
                
                fig_loadings2 = px.bar(
                    loadings_pc2, x='Loading PC2', y='Ativo',
                    title="🎯 Loadings - Segundo Componente", 
                    orientation='h',
                    color='Loading PC2',
                    color_continuous_scale='RdBu_r'
                )
                st.plotly_chart(fig_loadings2, use_container_width=True)
            
            # Interpretação dos componentes
            st.subheader("🧭 Interpretação dos Componentes")
            
            # PC1 interpretação
            dominant_pc1 = loadings_pc1.head(3)['Ativo'].tolist()
            st.info(f"**Primeiro Componente ({pca.explained_variance_ratio_[0]:.1%} da variância)**: "
                   f"Representa principalmente o movimento conjunto de {', '.join(dominant_pc1)}")
            
            # PC2 interpretação  
            dominant_pc2 = loadings_pc2.head(3)['Ativo'].tolist()
            st.info(f"**Segundo Componente ({pca.explained_variance_ratio_[1]:.1%} da variância)**: "
                   f"Representa principalmente o movimento conjunto de {', '.join(dominant_pc2)}")
        
        # Tab 2: PCA Rolling
        with pca_tabs[1]:
            st.subheader("📈 Análise PCA Rolling")
            st.info("💡 Analisa como os componentes principais evoluem ao longo do tempo")
            
            # Calcular PCA rolling
            rolling_variance = []
            rolling_loadings_pc1 = []
            rolling_dates = []
            
            with st.spinner("Calculando PCA rolling..."):
                for i in range(window_size, len(returns_selected)):
                    window_data = returns_selected.iloc[i-window_size:i]
                    
                    # PCA na janela
                    scaler_window = StandardScaler()
                    scaled_window = scaler_window.fit_transform(window_data.fillna(0))
                    pca_window = PCA(n_components=2, random_state=42)
                    pca_window.fit(scaled_window)
                    
                    rolling_variance.append(pca_window.explained_variance_ratio_[0])
                    rolling_loadings_pc1.append(pca_window.components_[0])
                    rolling_dates.append(returns_selected.index[i])
            
            # Plot variância explicada rolling
            fig_rolling_var = px.line(
                x=rolling_dates, 
                y=rolling_variance,
                title=f"📊 Variância Explicada do PC1 (Janela de {window_size} dias)",
                labels={'x': 'Data', 'y': 'Variância Explicada PC1'}
            )
            fig_rolling_var.update_layout(height=400)
            st.plotly_chart(fig_rolling_var, use_container_width=True)
            
            # Estabilidade dos loadings
            loadings_df = pd.DataFrame(rolling_loadings_pc1, 
                                     columns=selected_assets,
                                     index=rolling_dates)
            
            # Plot dos loadings principais ao longo do tempo
            main_assets = loadings_df.abs().mean().nlargest(4).index
            fig_loadings_time = go.Figure()
            
            for asset in main_assets:
                fig_loadings_time.add_trace(go.Scatter(
                    x=loadings_df.index,
                    y=loadings_df[asset],
                    name=asset,
                    mode='lines'
                ))
            
            fig_loadings_time.update_layout(
                title="🎯 Evolução dos Loadings Principais (PC1)",
                xaxis_title="Data",
                yaxis_title="Loading",
                height=400
            )
            st.plotly_chart(fig_loadings_time, use_container_width=True)
            
            # Métricas de estabilidade
            col1, col2, col3 = st.columns(3)
            with col1:
                var_stability = np.std(rolling_variance)
                st.metric("Estabilidade Variância", f"{var_stability:.4f}", 
                         help="Menor = mais estável")
            with col2:
                mean_var = np.mean(rolling_variance)
                st.metric("Variância Média PC1", f"{mean_var:.2%}")
            with col3:
                loading_stability = loadings_df.std().mean()
                st.metric("Estabilidade Loadings", f"{loading_stability:.4f}",
                         help="Menor = mais estável")
        
        # Tab 3: Seleção de Portfolio
        with pca_tabs[2]:
            st.subheader("🎯 Construção de Portfolio via PCA")
            st.info("💡 Use os componentes principais para construir portfolios otimizados")
            
            # Estratégias baseadas em PCA
            strategy_type = st.selectbox(
                "Estratégia de construção:",
                ["Maximum Diversification", "Minimum Variance", "Equal Risk Contribution"]
            )
            
            if st.button("🚀 Construir Portfolio PCA"):
                with st.spinner("Construindo portfolio..."):
                    # Usar componentes para construir pesos
                    if strategy_type == "Maximum Diversification":
                        # Pesos baseados no primeiro componente (invertidos para diversificação)
                        weights_raw = 1 / (abs(pca.components_[0]) + 0.001)
                        weights = weights_raw / weights_raw.sum()
                    elif strategy_type == "Minimum Variance":
                        # Pesos baseados na matriz de covariância dos componentes
                        cov_components = np.cov(components[:, :3].T)
                        inv_cov = np.linalg.pinv(cov_components)
                        weights_components = inv_cov @ np.ones(len(inv_cov)) / (np.ones(len(inv_cov)) @ inv_cov @ np.ones(len(inv_cov)))
                        weights = abs(pca.components_[:3].T @ weights_components)
                        weights = weights / weights.sum()
                    else:  # Equal Risk Contribution
                        # Pesos que equalizam a contribuição de risco
                        vol = returns_selected.std()
                        weights = (1 / vol) / (1 / vol).sum()
                    
                    # Construir portfolio
                    portfolio_returns = (returns_selected * weights).sum(axis=1)
                    portfolio_cumulative = (1 + portfolio_returns).cumprod()
                    
                    # Calcular métricas
                    total_return = portfolio_cumulative.iloc[-1] - 1
                    annual_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
                    annual_vol = portfolio_returns.std() * np.sqrt(252)
                    sharpe = annual_return / annual_vol if annual_vol > 0 else 0
                    
                    # Exibir resultados
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Retorno Total", f"{total_return:.2%}")
                    with col2:
                        st.metric("Retorno Anual", f"{annual_return:.2%}")
                    with col3:
                        st.metric("Volatilidade", f"{annual_vol:.2%}")
                    with col4:
                        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
                    
                    # Gráfico de alocação
                    weights_df = pd.DataFrame({
                        'Ativo': selected_assets,
                        'Peso (%)': weights * 100
                    }).sort_values('Peso (%)', ascending=False)
                    
                    fig_allocation = px.pie(
                        weights_df, values='Peso (%)', names='Ativo',
                        title=f"🎯 Alocação do Portfolio ({strategy_type})"
                    )
                    st.plotly_chart(fig_allocation, use_container_width=True)
                    
                    # Performance histórica
                    fig_performance = px.line(
                        x=portfolio_cumulative.index,
                        y=(portfolio_cumulative - 1) * 100,
                        title="📈 Performance do Portfolio PCA",
                        labels={'x': 'Data', 'y': 'Retorno Acumulado (%)'}
                    )
                    st.plotly_chart(fig_performance, use_container_width=True)
        
        # Tab 4: Análise de Risco
        with pca_tabs[3]:
            st.subheader("📉 Análise de Risco via PCA")
            st.info("💡 Use PCA para identificar os principais fatores de risco do portfolio")
            
            # Decomposição de risco por componente
            risk_contributions = []
            for i in range(min(5, n_components)):
                # Variância explicada por cada componente
                component_var = pca.explained_variance_[i]
                # Contribuição dos ativos para este componente
                loadings = abs(pca.components_[i])
                # Risco explicado por componente
                risk_contrib = component_var * loadings
                risk_contributions.append(risk_contrib)
            
            # Matriz de contribuições de risco
            risk_matrix = pd.DataFrame(
                risk_contributions,
                columns=selected_assets,
                index=[f'PC{i+1}' for i in range(len(risk_contributions))]
            )
            
            # Heatmap de risco
            fig_risk = px.imshow(
                risk_matrix.values,
                x=risk_matrix.columns,
                y=risk_matrix.index,
                color_continuous_scale='Reds',
                title="🔥 Mapa de Calor - Contribuição de Risco por Componente"
            )
            st.plotly_chart(fig_risk, use_container_width=True)
            
            # Top riscos
            total_risk_by_asset = risk_matrix.sum()
            top_risks = total_risk_by_asset.nlargest(5)
            
            st.subheader("⚠️ Principais Fatores de Risco")
            for i, (asset, risk) in enumerate(top_risks.items()):
                st.write(f"{i+1}. **{asset}**: Contribuição de risco = {risk:.4f}")
            
            # Diversificação via PCA
            effective_components = (pca.explained_variance_ratio_ ** 2).sum() ** -1
            st.metric("Número Efetivo de Componentes", f"{effective_components:.1f}",
                     help="Maior = mais diversificado")
        
        # Tab 5: Explicação Didática
        with pca_tabs[4]:
            st.subheader("🔬 Explicação Didática - PCA em Finanças")
            
            st.markdown("""
            ### 📚 Conceitos Fundamentais
            
            **1. O que é PCA?**
            - Análise de Componentes Principais reduz dados multidimensionais
            - Encontra direções de máxima variância nos dados
            - Cada componente é uma combinação linear dos ativos originais
            
            **2. Como interpretar os resultados?**
            
            #### 🎯 Variância Explicada
            - **PC1 com 60%**: O primeiro fator explica 60% da variação total do mercado
            - **PC1+PC2 com 80%**: Os dois primeiros fatores capturam 80% dos movimentos
            
            #### 📊 Loadings (Cargas)
            - **Loading positivo alto**: Ativo se move na mesma direção do componente
            - **Loading negativo alto**: Ativo se move na direção oposta
            - **Loading próximo de zero**: Ativo não está relacionado a este fator
            
            **3. Aplicações Práticas em Portfolios**
            
            #### 🎯 Construção de Portfolios
            - **Maximum Diversification**: Minimiza exposição ao fator dominante
            - **Factor Investing**: Investe especificamente em fatores identificados
            - **Risk Budgeting**: Aloca risco entre diferentes componentes
            
            #### 📉 Gestão de Risco
            - **Identificação de fatores**: Quais são os principais drivers de risco?
            - **Stress testing**: Como o portfolio reage a choques nos componentes principais?
            - **Hedging**: Construir hedges baseados nos fatores identificados
            
            **4. Limitações e Cuidados**
            
            ⚠️ **Estabilidade temporal**: Componentes podem mudar ao longo do tempo
            
            ⚠️ **Interpretação econômica**: Nem sempre os componentes têm significado econômico claro
            
            ⚠️ **Outliers**: Dados extremos podem distorcer os componentes
            
            ### 💡 Dicas Práticas
            
            - Use janelas rolling para capturar mudanças de regime
            - Combine PCA com conhecimento econômico/setorial
            - Monitore a estabilidade dos loadings
            - Considere re-estimar periodicamente
            """)
            
            # Exemplo interativo
            st.subheader("🔧 Exemplo Interativo")
            
            example_component = st.selectbox(
                "Selecione um componente para análise detalhada:",
                [f"PC{i+1}" for i in range(min(3, n_components))]
            )
            
            pc_idx = int(example_component[2:]) - 1
            
            # Análise detalhada do componente selecionado
            st.write(f"**{example_component}** explica {pca.explained_variance_ratio_[pc_idx]:.1%} da variância total")
            
            # Top contributors
            component_loadings = pd.DataFrame({
                'Ativo': selected_assets,
                'Loading': pca.components_[pc_idx],
                'Abs_Loading': abs(pca.components_[pc_idx])
            }).sort_values('Abs_Loading', ascending=False)
            
            st.write("**Principais contribuidores:**")
            for i, row in component_loadings.head(3).iterrows():
                direction = "positivamente" if row['Loading'] > 0 else "negativamente"
                st.write(f"- **{row['Ativo']}**: Contribui {direction} ({row['Loading']:.3f})")
                
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
        
        # Configurações na sidebar
        with st.sidebar:
            st.subheader("⚙️ Configurações Pair Trading")
            
            # Filtros de seleção
            min_correlation = st.slider("Correlação mínima:", 0.5, 0.95, 0.75, 0.05)
            min_data_points = st.slider("Mínimo de dados (dias):", 100, 1000, 252)
            
            # Parâmetros de trading
            st.subheader("📈 Parâmetros de Trading")
            entry_threshold = st.slider("Threshold de entrada (Z-Score):", 1.0, 3.0, 2.0, 0.1)
            exit_threshold = st.slider("Threshold de saída (Z-Score):", 0.1, 1.0, 0.5, 0.1)
            stop_loss = st.slider("Stop Loss (Z-Score):", 3.0, 5.0, 3.5, 0.1)
            
            # Custos de transação
            transaction_cost = st.slider("Custo de transação (%):", 0.0, 0.5, 0.1, 0.01)
        
        # Inicializar análise
        pair_analyzer = PairTradingAnalysis(df)
        
        # Tab 1: Identificar Pares
        with pair_tabs[0]:
            st.subheader("🔍 Identificação de Pares Cointegrados")
            
            # Seleção de ativos para análise
            all_assets = df.columns.tolist()
            selected_assets = st.multiselect(
                "Selecione ativos para busca de pares (deixe vazio para usar todos):",
                all_assets,
                default=[]
            )
            
            if not selected_assets:
                selected_assets = all_assets[:20]  # Limitar para performance
                st.info(f"📊 Usando os primeiros 20 ativos: {', '.join(selected_assets[:5])}...")
            
            if st.button("🔍 Buscar Pares Cointegrados"):
                with st.spinner("Analisando correlações e cointegração..."):
                    # Encontrar pares correlacionados
                    correlated_pairs = pair_analyzer.find_correlated_pairs(
                        min_correlation=min_correlation,
                        min_years=min_data_points/252
                    )
                    
                    if correlated_pairs:
                        st.success(f"✅ Encontrados {len(correlated_pairs)} pares altamente correlacionados!")
                        
                        # Testar cointegração
                        cointegrated_pairs = []
                        progress_bar = st.progress(0)
                        
                        for i, pair in enumerate(correlated_pairs[:20]):  # Limitar para performance
                            coint_result = pair_analyzer.test_cointegration(
                                pair['asset1'], pair['asset2']
                            )
                            
                            if coint_result and coint_result.get('is_cointegrated', False):
                                cointegrated_pairs.append({
                                    **pair,
                                    'p_value': coint_result['p_value'],
                                    'cointegration_stat': coint_result['cointegration_stat']
                                })
                            
                            progress_bar.progress((i + 1) / min(20, len(correlated_pairs)))
                        
                        if cointegrated_pairs:
                            st.success(f"🎯 Encontrados {len(cointegrated_pairs)} pares cointegrados!")
                            
                            # Tabela de resultados
                            pairs_df = pd.DataFrame(cointegrated_pairs)
                            pairs_df = pairs_df.sort_values('correlation', key=abs, ascending=False)
                            
                            # Formatação da tabela
                            display_df = pairs_df.copy()
                            display_df['correlation'] = display_df['correlation'].apply(lambda x: f"{x:.3f}")
                            display_df['p_value'] = display_df['p_value'].apply(lambda x: f"{x:.4f}")
                            display_df['cointegration_stat'] = display_df['cointegration_stat'].apply(lambda x: f"{x:.3f}")
                            
                            st.dataframe(
                                display_df[['asset1', 'asset2', 'correlation', 'p_value', 'cointegration_stat']],
                                column_config={
                                    'asset1': 'Ativo 1',
                                    'asset2': 'Ativo 2', 
                                    'correlation': 'Correlação',
                                    'p_value': 'P-Value Coint.',
                                    'cointegration_stat': 'Stat Coint.'
                                },
                                use_container_width=True
                            )
                            
                            # Salvar no session_state
                            st.session_state['cointegrated_pairs'] = cointegrated_pairs
                            
                        else:
                            st.warning("⚠️ Nenhum par cointegrado encontrado com os critérios atuais.")
                            st.info("💡 Tente reduzir a correlação mínima ou aumentar o período de dados.")
                    else:
                        st.warning("⚠️ Nenhum par com correlação suficiente encontrado.")
                        
            # Matriz de correlação visual
            if st.checkbox("📊 Mostrar Matriz de Correlação"):
                if len(selected_assets) <= 15:  # Evitar matriz muito grande
                    correlation_matrix = df[selected_assets].corr()
                    
                    fig_corr = px.imshow(
                        correlation_matrix,
                        title="🔗 Matriz de Correlação",
                        color_continuous_scale='RdBu_r',
                        aspect="auto"
                    )
                    fig_corr.update_layout(height=600)
                    st.plotly_chart(fig_corr, use_container_width=True)
                else:
                    st.info("📊 Muitos ativos selecionados. Matriz não exibida para melhor performance.")
          # Tab 2: Análise Detalhada
        with pair_tabs[1]:
            st.subheader("📊 Análise Detalhada do Par")
            
            # Seleção manual ou automática
            analysis_mode = st.radio(
                "Modo de análise:",
                ["🎯 Selecionar par automaticamente", "✋ Selecionar par manualmente"]
            )
            
            asset1, asset2 = None, None
            
            if analysis_mode == "✋ Selecionar par manualmente":
                col1, col2 = st.columns(2)
                with col1:
                    asset1 = st.selectbox("Primeiro ativo:", df.columns.tolist(), key="manual_asset1")
                with col2:
                    asset2 = st.selectbox("Segundo ativo:", df.columns.tolist(), key="manual_asset2")
            else:
                # Usar melhor par encontrado
                if 'cointegrated_pairs' in st.session_state and st.session_state['cointegrated_pairs']:
                    best_pair = st.session_state['cointegrated_pairs'][0]
                    asset1, asset2 = best_pair['asset1'], best_pair['asset2']
                    st.info(f"🏆 Analisando melhor par: **{asset1}** vs **{asset2}**")
                else:
                    st.warning("⚠️ Execute a busca de pares primeiro na aba 'Identificar Pares'")
                    asset1, asset2 = df.columns[0], df.columns[1]
            
            if asset1 and asset2 and asset1 != asset2:
                # Executar análise detalhada
                coint_result = pair_analyzer.test_cointegration(asset1, asset2)
                
                if coint_result:
                    # Métricas principais
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        # Verificar se a matriz de correlação existe e se os ativos estão nela
                        try:
                            if (pair_analyzer.correlation_matrix is not None and 
                                not pair_analyzer.correlation_matrix.empty and
                                asset1 in pair_analyzer.correlation_matrix.index and
                                asset2 in pair_analyzer.correlation_matrix.columns):
                                correlation_value = pair_analyzer.correlation_matrix.loc[asset1, asset2]
                                if not pd.isna(correlation_value):
                                    st.metric("Correlação", f"{correlation_value:.3f}")
                                else:
                                    st.metric("Correlação", "N/A")
                            else:
                                st.metric("Correlação", "N/A")
                        except Exception as e:
                            st.metric("Correlação", "Erro")
                            st.error(f"Erro ao calcular correlação: {str(e)}")
                    with col2:
                        status = "✅ Sim" if coint_result.get('is_cointegrated', False) else "❌ Não"
                        st.metric("Cointegrado", status)
                    with col3:
                        st.metric("P-Value", f"{coint_result.get('p_value', 0):.4f}")
                    with col4:
                        st.metric("R²", f"{coint_result.get('r_squared', 0):.3f}")
                    
                    # Gráficos de preços e spread
                    price1 = coint_result['price1']
                    price2 = coint_result['price2'] 
                    spread = coint_result['spread']
                    
                    # Normalizar preços para comparação
                    price1_norm = price1 / price1.iloc[0]
                    price2_norm = price2 / price2.iloc[0]
                    
                    # Plot preços normalizados
                    fig_prices = go.Figure()
                    fig_prices.add_trace(go.Scatter(
                        x=price1_norm.index, y=price1_norm.values,
                        name=asset1, line=dict(color='blue')
                    ))
                    fig_prices.add_trace(go.Scatter(
                        x=price2_norm.index, y=price2_norm.values,
                        name=asset2, line=dict(color='red')
                    ))
                    fig_prices.update_layout(
                        title=f"📈 Preços Normalizados: {asset1} vs {asset2}",
                        xaxis_title="Data",
                        yaxis_title="Preço Normalizado",
                        height=400
                    )
                    st.plotly_chart(fig_prices, use_container_width=True)
                    
                    # Plot do spread
                    spread_mean = coint_result['spread_mean']
                    spread_std = coint_result['spread_std']
                    z_score = (spread - spread_mean) / spread_std
                    
                    fig_spread = make_subplots(
                        rows=2, cols=1,
                        subplot_titles=("Spread", "Z-Score"),
                        vertical_spacing=0.1
                    )
                    
                    # Spread
                    fig_spread.add_trace(go.Scatter(
                        x=spread.index, y=spread.values,
                        name='Spread', line=dict(color='green')
                    ), row=1, col=1)
                    fig_spread.add_hline(y=spread_mean, line_dash="dash", line_color="gray", row=1, col=1)
                    
                    # Z-Score
                    fig_spread.add_trace(go.Scatter(
                        x=z_score.index, y=z_score.values,
                        name='Z-Score', line=dict(color='orange')
                    ), row=2, col=1)
                    
                    # Linhas de threshold
                    fig_spread.add_hline(y=entry_threshold, line_dash="dash", line_color="red", row=2, col=1)
                    fig_spread.add_hline(y=-entry_threshold, line_dash="dash", line_color="red", row=2, col=1)
                    fig_spread.add_hline(y=exit_threshold, line_dash="dot", line_color="blue", row=2, col=1)
                    fig_spread.add_hline(y=-exit_threshold, line_dash="dot", line_color="blue", row=2, col=1)
                    fig_spread.add_hline(y=0, line_dash="solid", line_color="black", row=2, col=1)
                    
                    fig_spread.update_layout(
                        title=f"📊 Análise do Spread: {asset1} vs {asset2}",
                        height=600
                    )
                    st.plotly_chart(fig_spread, use_container_width=True)
                    
                    # Estatísticas do spread
                    st.subheader("📈 Estatísticas do Spread")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Média", f"{spread_mean:.4f}")
                    with col2:
                        st.metric("Desvio Padrão", f"{spread_std:.4f}")
                    with col3:
                        current_z = z_score.iloc[-1]
                        st.metric("Z-Score Atual", f"{current_z:.2f}")
                    with col4:
                        # Sinal atual
                        if abs(current_z) > entry_threshold:
                            signal = "🔴 ENTRADA" if current_z > 0 else "🟢 ENTRADA"
                        elif abs(current_z) < exit_threshold:
                            signal = "🔵 SAÍDA"
                        else:
                            signal = "⚪ NEUTRO"
                        st.metric("Sinal Atual", signal)
        
        # Tab 3: Sinais de Trading
        with pair_tabs[2]:
            st.subheader("⚡ Sinais de Trading")
            
            if asset1 and asset2 and asset1 != asset2:
                # Gerar sinais de trading
                coint_result = pair_analyzer.test_cointegration(asset1, asset2)
                
                if coint_result:
                    signals = pair_analyzer.generate_trading_signals(
                        coint_result, 
                        entry_threshold=entry_threshold,
                        exit_threshold=exit_threshold
                    )
                    
                    # Estatísticas dos sinais
                    buy_signals = len(signals[signals['signal'] == 1])
                    sell_signals = len(signals[signals['signal'] == -1])
                    total_signals = buy_signals + sell_signals
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Sinais de Compra", buy_signals)
                    with col2:
                        st.metric("Sinais de Venda", sell_signals)
                    with col3:
                        st.metric("Total de Sinais", total_signals)
                    with col4:
                        freq = f"{total_signals / (len(signals) / 252):.1f}" if len(signals) > 252 else "N/A"
                        st.metric("Sinais/Ano", freq)
                    
                    # Visualização dos sinais
                    fig_signals = go.Figure()
                    
                    # Z-Score
                    fig_signals.add_trace(go.Scatter(
                        x=signals.index, y=signals['z_score'],
                        name='Z-Score', line=dict(color='gray')
                    ))
                    
                    # Sinais de compra (quando z_score < -threshold)
                    buy_points = signals[signals['signal'] == 1]
                    if len(buy_points) > 0:
                        fig_signals.add_trace(go.Scatter(
                            x=buy_points.index, y=buy_points['z_score'],
                            mode='markers', name='Sinal Compra',
                            marker=dict(color='green', size=10, symbol='triangle-up')
                        ))
                    
                    # Sinais de venda (quando z_score > threshold)
                    sell_points = signals[signals['signal'] == -1]
                    if len(sell_points) > 0:
                        fig_signals.add_trace(go.Scatter(
                            x=sell_points.index, y=sell_points['z_score'],
                            mode='markers', name='Sinal Venda',
                            marker=dict(color='red', size=10, symbol='triangle-down')
                        ))
                    
                    # Linhas de threshold
                    fig_signals.add_hline(y=entry_threshold, line_dash="dash", line_color="red")
                    fig_signals.add_hline(y=-entry_threshold, line_dash="dash", line_color="red")
                    fig_signals.add_hline(y=exit_threshold, line_dash="dot", line_color="blue")
                    fig_signals.add_hline(y=-exit_threshold, line_dash="dot", line_color="blue")
                    fig_signals.add_hline(y=0, line_dash="solid", line_color="black")
                    
                    fig_signals.update_layout(
                        title=f"⚡ Sinais de Trading: {asset1} vs {asset2}",
                        xaxis_title="Data",
                        yaxis_title="Z-Score",
                        height=500
                    )
                    st.plotly_chart(fig_signals, use_container_width=True)
                    
                    # Tabela de sinais recentes
                    recent_signals = signals[signals['signal'] != 0].tail(10)
                    if len(recent_signals) > 0:
                        st.subheader("🕐 Sinais Recentes")
                        
                        # Formatação da tabela
                        display_signals = recent_signals.copy()
                        display_signals['Tipo'] = display_signals['signal'].apply(
                            lambda x: "🟢 Compra" if x == 1 else "🔴 Venda"
                        )
                        display_signals['Z-Score'] = display_signals['z_score'].apply(lambda x: f"{x:.2f}")
                        display_signals['Data'] = display_signals.index.strftime('%Y-%m-%d')
                        
                        st.dataframe(
                            display_signals[['Data', 'Tipo', 'Z-Score']],
                            use_container_width=True
                        )
            else:
                st.info("📊 Selecione um par válido na aba 'Análise Detalhada' primeiro.")
        
        # Tab 4: Backtest
        with pair_tabs[3]:
            st.subheader("📈 Backtest da Estratégia")
            
            if asset1 and asset2 and asset1 != asset2:
                # Configurações do backtest
                col1, col2 = st.columns(2)
                with col1:
                    initial_capital = st.number_input(
                        "Capital inicial (R$):", 
                        min_value=1000, max_value=1000000, 
                        value=100000, step=1000
                    )
                with col2:
                    lookback_period = st.slider(
                        "Período de lookback (dias):", 
                        30, 252, 90
                    )
                
                if st.button("🚀 Executar Backtest"):
                    with st.spinner("Executando backtest..."):
                        # Executar análise completa
                        coint_result = pair_analyzer.test_cointegration(asset1, asset2)
                        
                        if coint_result:
                            signals = pair_analyzer.generate_trading_signals(
                                coint_result,
                                entry_threshold=entry_threshold,
                                exit_threshold=exit_threshold
                            )
                            
                            backtest_result = pair_analyzer.backtest_strategy(
                                coint_result, signals, 
                                transaction_cost=transaction_cost/100
                            )
                            
                            if backtest_result:
                                # Métricas de performance
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Retorno Total", f"{backtest_result['total_return']:.2%}")
                                with col2:
                                    st.metric("Retorno Anual", f"{backtest_result['annual_return']:.2%}")
                                with col3:
                                    st.metric("Sharpe Ratio", f"{backtest_result['sharpe_ratio']:.2f}")
                                with col4:
                                    st.metric("Max Drawdown", f"{backtest_result['max_drawdown']:.2%}")
                                
                                # Segunda linha de métricas
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("Volatilidade", f"{backtest_result['annual_volatility']:.2%}")
                                with col2:
                                    st.metric("Número de Trades", f"{backtest_result['num_trades']}")
                                with col3:
                                    win_rate = backtest_result.get('win_rate', 0)
                                    st.metric("Win Rate", f"{win_rate:.1%}")
                                with col4:
                                    capital_final = initial_capital * (1 + backtest_result['total_return'])
                                    st.metric("Capital Final", f"R$ {capital_final:,.0f}")
                                
                                # Gráfico de performance
                                equity_curve = backtest_result['cumulative_returns'] * initial_capital
                                
                                fig_performance = go.Figure()
                                fig_performance.add_trace(go.Scatter(
                                    x=equity_curve.index, y=equity_curve.values,
                                    name='Portfolio Pair Trading',
                                    line=dict(color='green', width=2)
                                ))
                                
                                # Comparar com buy & hold dos ativos individuais
                                price1_norm = coint_result['price1'] / coint_result['price1'].iloc[0] * initial_capital
                                price2_norm = coint_result['price2'] / coint_result['price2'].iloc[0] * initial_capital
                                
                                fig_performance.add_trace(go.Scatter(
                                    x=price1_norm.index, y=price1_norm.values,
                                    name=f'Buy & Hold {asset1}',
                                    line=dict(color='blue', dash='dash')
                                ))
                                
                                fig_performance.add_trace(go.Scatter(
                                    x=price2_norm.index, y=price2_norm.values,
                                    name=f'Buy & Hold {asset2}',
                                    line=dict(color='red', dash='dash')
                                ))
                                
                                fig_performance.update_layout(
                                    title=f"📈 Performance do Backtest: {asset1} vs {asset2}",
                                    xaxis_title="Data",
                                    yaxis_title="Capital (R$)",
                                    height=500
                                )
                                st.plotly_chart(fig_performance, use_container_width=True)
                                
                                # Interpretação dos resultados
                                st.subheader("🎯 Interpretação dos Resultados")
                                
                                if backtest_result['sharpe_ratio'] > 1.0:
                                    st.success("✅ **Estratégia Promissora**: Sharpe Ratio > 1.0 indica boa relação risco-retorno")
                                elif backtest_result['sharpe_ratio'] > 0.5:
                                    st.info("💡 **Estratégia Moderada**: Performance razoável, mas pode ser melhorada")
                                else:
                                    st.warning("⚠️ **Estratégia Questionável**: Baixo Sharpe Ratio indica pouco retorno pelo risco")
                                
                                if backtest_result['total_return'] > 0:
                                    st.info(f"📈 A estratégia gerou {backtest_result['total_return']:.1%} de retorno no período")
                                else:
                                    st.warning(f"📉 A estratégia teve prejuízo de {abs(backtest_result['total_return']):.1%}")
                                
                            else:
                                st.error("❌ Erro no backtest")
                        else:
                            st.error("❌ Erro na análise de cointegração")
            else:
                st.info("📊 Selecione um par válido na aba 'Análise Detalhada' primeiro.")
        
        # Tab 5: Otimização
        with pair_tabs[4]:
            st.subheader("🎯 Otimização de Parâmetros")
            st.info("💡 Encontre os melhores parâmetros para maximizar o Sharpe Ratio")
            
            if asset1 and asset2 and asset1 != asset2:
                # Configurar ranges de otimização
                col1, col2 = st.columns(2)
                with col1:
                    entry_range = st.slider(
                        "Range threshold entrada:", 
                        1.0, 4.0, (1.5, 3.0), 0.1
                    )
                with col2:
                    exit_range = st.slider(
                        "Range threshold saída:", 
                        0.1, 1.5, (0.2, 1.0), 0.1
                    )
                
                optimization_step = st.slider("Passo da otimização:", 0.1, 0.5, 0.2, 0.1)
                
                if st.button("🔍 Otimizar Parâmetros"):
                    with st.spinner("Otimizando parâmetros..."):
                        # Grid search nos parâmetros
                        best_sharpe = -np.inf
                        best_params = {}
                        optimization_results = []
                        
                        entry_values = np.arange(entry_range[0], entry_range[1] + optimization_step, optimization_step)
                        exit_values = np.arange(exit_range[0], exit_range[1] + optimization_step, optimization_step)
                        
                        total_combinations = len(entry_values) * len(exit_values)
                        progress_bar = st.progress(0)
                        current_combination = 0
                        
                        coint_result = pair_analyzer.test_cointegration(asset1, asset2)
                        
                        for entry_thresh in entry_values:
                            for exit_thresh in exit_values:
                                if exit_thresh < entry_thresh:  # Exit deve ser menor que entry
                                    # Gerar sinais
                                    signals = pair_analyzer.generate_trading_signals(
                                        coint_result,
                                        entry_threshold=entry_thresh,
                                        exit_threshold=exit_thresh
                                    )
                                    
                                    # Backtest
                                    backtest = pair_analyzer.backtest_strategy(
                                        coint_result, signals,
                                        transaction_cost=transaction_cost/100
                                    )
                                    
                                    if backtest:
                                        optimization_results.append({
                                            'entry_threshold': entry_thresh,
                                            'exit_threshold': exit_thresh,
                                            'sharpe_ratio': backtest['sharpe_ratio'],
                                            'total_return': backtest['total_return'],
                                            'max_drawdown': backtest['max_drawdown'],
                                            'num_trades': backtest['num_trades']
                                        })
                                        
                                        if backtest['sharpe_ratio'] > best_sharpe:
                                            best_sharpe = backtest['sharpe_ratio']
                                            best_params = {
                                                'entry_threshold': entry_thresh,
                                                'exit_threshold': exit_thresh,
                                                'backtest': backtest
                                            }
                                
                                current_combination += 1
                                progress_bar.progress(current_combination / total_combinations)
                        
                        if best_params:
                            st.success("🏆 Otimização concluída!")
                            
                            # Melhores parâmetros
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Melhor Entry", f"{best_params['entry_threshold']:.1f}")
                            with col2:
                                st.metric("Melhor Exit", f"{best_params['exit_threshold']:.1f}")
                            with col3:
                                st.metric("Sharpe Otimizado", f"{best_sharpe:.2f}")
                            with col4:
                                st.metric("Retorno Otimizado", f"{best_params['backtest']['total_return']:.2%}")
                            
                            # Heatmap dos resultados
                            if optimization_results:
                                opt_df = pd.DataFrame(optimization_results)
                                
                                # Criar matriz para heatmap
                                heatmap_data = opt_df.pivot(
                                    index='entry_threshold', 
                                    columns='exit_threshold', 
                                    values='sharpe_ratio'
                                )
                                
                                fig_heatmap = px.imshow(
                                    heatmap_data,
                                    title="🔥 Heatmap de Otimização - Sharpe Ratio",
                                    labels={'x': 'Exit Threshold', 'y': 'Entry Threshold'},
                                    color_continuous_scale='RdYlBu_r'
                                )
                                st.plotly_chart(fig_heatmap, use_container_width=True)
                                
                                # Top 5 combinações
                                st.subheader("🏆 Top 5 Combinações")
                                top_combinations = opt_df.nlargest(5, 'sharpe_ratio')
                                st.dataframe(
                                    top_combinations.round(4),
                                    use_container_width=True
                                )
                        else:
                            st.warning("⚠️ Nenhuma combinação válida encontrada")
            else:
                st.info("📊 Selecione um par válido na aba 'Análise Detalhada' primeiro.")
        
        # Tab 6: Tutorial
        with pair_tabs[5]:
            st.subheader("📚 Tutorial Completo - Pair Trading")
            
            st.markdown("""
            ### 🎯 O que é Pair Trading?
            
            **Pair Trading** é uma estratégia **market-neutral** que busca lucrar com divergências temporárias 
            entre ativos que historicamente se movem juntos.
            
            #### 🔍 Conceitos Fundamentais
            
            **1. Cointegração**
            - Dois ativos são cointegrados se existe uma relação de **longo prazo** entre eles
            - Mesmo que os preços divirjam temporariamente, tendem a **convergir** eventualmente
            - Teste estatístico: p-value < 0.05 indica cointegração significativa
            
            **2. Spread**
            - **Spread = Preço Ativo A - β × Preço Ativo B**
            - β (beta) é o **hedge ratio** que minimiza a variância do spread
            - Spread estacionário é essencial para a estratégia funcionar
            
            **3. Z-Score**
            - **Z-Score = (Spread - Média) / Desvio Padrão**
            - Mede quantos desvios padrão o spread está da média
            - Base para sinais de entrada e saída
            
            ### ⚡ Como Funciona a Estratégia?
            
            #### 📈 Sinais de Entrada
            - **Z-Score > +2.0**: Spread muito alto → **VENDER** spread (short A, long B)
            - **Z-Score < -2.0**: Spread muito baixo → **COMPRAR** spread (long A, short B)
            
            #### 📉 Sinais de Saída
            - **|Z-Score| < 0.5**: Spread voltou ao normal → **FECHAR** posição
            - **Stop Loss**: Z-Score > 3.5 → Sair para limitar perdas
            
            ### 🎛️ Parâmetros Importantes
            
            **Entry Threshold (2.0)**
            - Maior = menos sinais, mais seletivo
            - Menor = mais sinais, mais agressivo
            
            **Exit Threshold (0.5)**
            - Maior = sair mais cedo, menos ganho por trade
            - Menor = sair mais tarde, mais ganho mas mais risco
            
            **Stop Loss (3.5)**
            - Proteção contra divergências permanentes
            - Evita perdas catastróficas
            
            ### 📊 Métricas de Avaliação
            
            **Sharpe Ratio**
            - > 2.0: Excelente
            - 1.0-2.0: Bom
            - 0.5-1.0: Moderado
            - < 0.5: Questionável
            
            **Win Rate**
            - Taxa de trades lucrativos
            - 60%+ é considerado bom
            
            **Maximum Drawdown**
            - Maior perda de pico a vale
            - < 10% é preferível
            
            ### ⚠️ Riscos e Limitações
            
            **1. Quebra de Cointegração**
            - Mudanças estruturais podem quebrar a relação histórica
            - **Solução**: Monitorar regularmente e re-testar cointegração
            
            **2. Regime Changes**
            - Crises podem alterar correlações temporariamente
            - **Solução**: Stop losses e análise de contexto macro
            
            **3. Custos de Transação**
            - Alta frequência de trades pode corroer retornos
            - **Solução**: Otimizar parâmetros considerando custos
            
            **4. Execução**
            - Dificuldade em executar trades simultâneos
            - **Solução**: Usar spreads ou ETFs quando possível
            
            ### 💡 Dicas Práticas
            
            ✅ **Escolha ativos do mesmo setor** (bancos, varejo, etc.)
            
            ✅ **Use dados de pelo menos 2 anos** para testar cointegração
            
            ✅ **Monitore correlação rolling** para detectar mudanças
            
            ✅ **Considere fatores fundamentais** além de estatísticas
            
            ✅ **Diversifique entre múltiplos pares** para reduzir risco
            
            ✅ **Re-otimize parâmetros periodicamente** (trimestral/semestral)
            
            ### 🚀 Próximos Passos
            
            1. **Identificar Pares**: Use a aba "Identificar Pares" para encontrar candidatos
            2. **Análise Detalhada**: Valide cointegração e examine o spread
            3. **Sinais**: Configure thresholds apropriados para seu perfil de risco
            4. **Backtest**: Teste a estratégia em dados históricos
            5. **Otimização**: Encontre parâmetros ótimos
            6. **Implementação**: Execute com capital real (começe pequeno!)
            
            **Lembre-se**: Pair Trading requer disciplina, paciência e gestão de risco rigorosa!
            """)
            
    except Exception as e:
        st.error(f"❌ Erro ao carregar dados: {str(e)}")

def create_risk_metrics_analysis(pca_metrics, mt5_data):
    """Cria análise detalhada de métricas de risco"""
    try:
        # Extrair dados MT5
        mt5_drawdown = float(mt5_data.get('drawdown', '0%').replace('%', '')) / 100
        mt5_profit_factor = mt5_data.get('profit_factor', 0)
        mt5_recovery_factor = mt5_data.get('recovery_factor', 0)
        mt5_win_rate = mt5_data.get('win_rate', 0) / 100
        
        # Calcular métricas PCA adicionais
        risk_free_rate = 0.1075  # Selic
        pca_sharpe = (pca_metrics['annual_return'] - risk_free_rate) / pca_metrics['annual_volatility'] if pca_metrics['annual_volatility'] > 0 else 0
        
        # Fator de recuperação PCA (retorno anual / max drawdown)
        pca_recovery = abs(pca_metrics['annual_return'] / pca_metrics['max_drawdown']) if pca_metrics['max_drawdown'] != 0 else 0
        
        return {
            'pca': {
                'sharpe_ratio': pca_sharpe,
                'recovery_factor': pca_recovery,
                'calmar_ratio': pca_metrics['annual_return'] / abs(pca_metrics['max_drawdown']) if pca_metrics['max_drawdown'] != 0 else 0,
                'volatility': pca_metrics['annual_volatility'],
                'max_drawdown': abs(pca_metrics['max_drawdown'])
            },
            'mt5': {
                'profit_factor': mt5_profit_factor,
                'recovery_factor': mt5_recovery_factor,
                'win_rate': mt5_win_rate,
                'drawdown': mt5_drawdown
            }
        }
    except Exception as e:
        st.error(f"Erro ao calcular métricas de risco: {e}")
        return None

def create_performance_radar_chart(pca_metrics, mt5_data, risk_metrics):
    """Cria gráfico radar comparando múltiplas dimensões"""
    try:
        # Normalizar métricas para escala 0-10
        def normalize_metric(value, min_val, max_val):
            if max_val == min_val:
                return 5
            return max(0, min(10, (value - min_val) / (max_val - min_val) * 10))
        
        categories = ['Retorno', 'Risco (inv)', 'Sharpe', 'Recuperação', 'Consistência']
        
        # PCA normalizado
        pca_values = [
            normalize_metric(pca_metrics['annual_return'], -0.5, 1.0),  # Retorno
            normalize_metric(1 / max(pca_metrics['annual_volatility'], 0.01), 0, 10),  # Risco invertido
            normalize_metric(risk_metrics['pca']['sharpe_ratio'], -2, 3),  # Sharpe
            normalize_metric(risk_metrics['pca']['recovery_factor'], 0, 10),  # Recuperação
            normalize_metric(1 / max(abs(pca_metrics['max_drawdown']), 0.01), 0, 10)  # Consistência
        ]
        
        # MT5 normalizado
        mt5_return = mt5_data.get('net_profit', 0) / max(mt5_data.get('initial_capital', 1), 1)
        mt5_values = [
            normalize_metric(mt5_return, -0.5, 1.0),  # Retorno
            normalize_metric(1 / max(risk_metrics['mt5']['drawdown'], 0.01), 0, 10),  # Risco invertido
            normalize_metric(risk_metrics['mt5']['profit_factor'], 0, 3),  # Profit Factor como proxy Sharpe
            normalize_metric(risk_metrics['mt5']['recovery_factor'], 0, 10),  # Recuperação
            normalize_metric(risk_metrics['mt5']['win_rate'], 0, 1) * 10  # Win Rate como consistência
        ]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=pca_values,
            theta=categories,
            fill='toself',
            name=PCA_PORTFOLIO_LABEL,
            line_color='blue',
            fillcolor='rgba(0,0,255,0.1)'
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=mt5_values,
            theta=categories,
            fill='toself',
            name=MT5_REAL_LABEL,
            line_color='red',
            fillcolor='rgba(255,0,0,0.1)'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 10]
                )
            ),
            title="Análise Multidimensional de Performance",
            showlegend=True,
            height=500
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Erro ao criar gráfico radar: {e}")
        return None

def plot_temporal_comparison(pca_returns, mt5_data, initial_capital):
    """Plota comparação temporal entre PCA e MT5"""
    try:
        # Calcular curva de equity PCA
        pca_cum = (1 + pca_returns).cumprod() * initial_capital
        
        # Extrair dados MT5
        mt5_balance = mt5_data.get('balance', initial_capital)
        mt5_initial = mt5_data.get('initial_capital', initial_capital)
        
        # Criar curva simulada MT5 (interpolação linear)
        dates = pca_cum.index
        mt5_values = np.linspace(mt5_initial, mt5_balance, len(dates))
        mt5_curve = pd.Series(mt5_values, index=dates, name=MT5_REAL_LABEL)
        
        # Criar gráfico interativo com Plotly
        fig = go.Figure()
        
        # Adicionar linha PCA
        fig.add_trace(go.Scatter(
            x=pca_cum.index,
            y=pca_cum.values,
            mode='lines',
            name=PCA_PORTFOLIO_LABEL,
            line=dict(color='blue', width=2),
            hovertemplate='<b>PCA</b><br>Data: %{x}<br>Equity: R$ %{y:,.2f}<extra></extra>'
        ))
        
        # Adicionar linha MT5
        fig.add_trace(go.Scatter(
            x=mt5_curve.index,
            y=mt5_curve.values,
            mode='lines',
            name=MT5_REAL_LABEL,
            line=dict(color='red', width=2, dash='dash'),
            hovertemplate='<b>MT5</b><br>Data: %{x}<br>Equity: R$ %{y:,.2f}<extra></extra>'
        ))
        
        # Configurar layout
        fig.update_layout(
            title='Evolução Temporal das Estratégias',
            xaxis_title='Data',
            yaxis_title='Equity (R$)',
            hovermode='x unified',
            showlegend=True,
            height=500
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Erro ao criar gráfico temporal: {str(e)}")
        return None

def plot_drawdown_comparison(pca_returns, mt5_data):
    """Compara drawdown entre PCA e MT5"""
    try:
        # Calcular drawdown PCA
        pca_cum = (1 + pca_returns).cumprod()
        running_max = pca_cum.expanding().max()
        pca_drawdown = (pca_cum / running_max - 1) * 100
        
        # Extrair drawdown MT5
        mt5_dd_str = mt5_data.get('drawdown', '0%')
        mt5_dd_value = float(mt5_dd_str.replace('%', ''))
        
        # Criar série temporal simulada para MT5 drawdown
        dates = pca_drawdown.index
        rng = np.random.default_rng(42)
        mt5_dd_values = rng.uniform(-mt5_dd_value, 0, len(dates))
        mt5_drawdown = pd.Series(mt5_dd_values, index=dates).rolling(window=10).mean()
        
        fig = go.Figure()
        
        # Drawdown PCA
        fig.add_trace(go.Scatter(
            x=pca_drawdown.index,
            y=pca_drawdown.values,
            mode='lines',
            name='PCA Drawdown',
            fill='tonexty',
            fillcolor='rgba(255,0,0,0.3)',
            line=dict(color='red'),
            hovertemplate='<b>PCA DD</b><br>%{y:.2f}%<extra></extra>'
        ))
        
        # Drawdown MT5
        fig.add_trace(go.Scatter(
            x=mt5_drawdown.index,
            y=mt5_drawdown.values,
            mode='lines',
            name='MT5 Drawdown',
            line=dict(color='orange', dash='dash'),
            hovertemplate='<b>MT5 DD</b><br>%{y:.2f}%<extra></extra>'
        ))
        
        fig.update_layout(
            title='Comparação de Drawdown',
            xaxis_title='Data',
            yaxis_title='Drawdown (%)',
            height=400
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Erro ao criar gráfico drawdown: {str(e)}")
        return None

def create_portfolio_allocation_analysis(mt5_data):
    """Analisa alocação de símbolos do MT5"""
    try:
        symbols = mt5_data.get('symbols', {})
        if not symbols:
            return None
            
        # Filtrar símbolos com valores válidos
        valid_symbols = {k: v for k, v in symbols.items() if isinstance(v, (int, float)) and v != 0}
        
        if not valid_symbols:
            return None
            
        # Criar DataFrame
        symbol_df = pd.DataFrame([
            {SYMBOL_COLUMN: symbol, 'P&L': profit, PL_ABS_COLUMN: abs(profit)}
            for symbol, profit in valid_symbols.items()
        ])
        
        # Calcular percentuais
        total_abs = symbol_df[PL_ABS_COLUMN].sum()
        if total_abs > 0:
            symbol_df['Percentual'] = (symbol_df[PL_ABS_COLUMN] / total_abs * 100).round(2)
        else:
            symbol_df['Percentual'] = 0
            
        symbol_df['Tipo'] = symbol_df['P&L'].apply(lambda x: PROFIT_LABEL if x > 0 else LOSS_LABEL)
        
        # Gráfico de pizza
        fig_pie = px.pie(
            symbol_df, 
            values=PL_ABS_COLUMN, 
            names=SYMBOL_COLUMN,
            color='Tipo',
            color_discrete_map={PROFIT_LABEL: 'green', LOSS_LABEL: 'red'},
            title='Distribuição de P&L por Símbolo'
        )
        
        # Gráfico de barras
        fig_bar = px.bar(
            symbol_df.sort_values('P&L'), 
            x='P&L', 
            y=SYMBOL_COLUMN,
            color='Tipo',
            color_discrete_map={PROFIT_LABEL: 'green', LOSS_LABEL: 'red'},
            title='P&L por Símbolo',
            orientation='h'
        )
        
        return {'pie': fig_pie, 'bar': fig_bar, 'data': symbol_df}
        
    except Exception as e:
        st.error(f"Erro ao analisar alocação: {e}")
        return None

# =====================================================================
# NAVEGAÇÃO PRINCIPAL
# =====================================================================

def main():
    """Função principal da aplicação"""
    
    # Detectar tema atual do Streamlit
    st.markdown("""
    <script>
    // Detectar tema do Streamlit
    function getStreamlitTheme() {
        const rootElement = document.querySelector('.stApp');
        if (rootElement) {
            const computedStyle = getComputedStyle(rootElement);
            const bgColor = computedStyle.backgroundColor;
            // Se o background for escuro, estamos no dark mode
            const rgb = bgColor.match(/\\d+/g);
            if (rgb) {
                const brightness = (parseInt(rgb[0]) * 299 + parseInt(rgb[1]) * 587 + parseInt(rgb[2]) * 114) / 1000;
                return brightness < 128 ? 'dark' : 'light';
            }
        }
        return 'light';
    }
    
    // Aplicar tema dinâmico
    function applyTheme() {
        const theme = getStreamlitTheme();
        document.documentElement.setAttribute('data-theme', theme);
    }
    
    // Observar mudanças no tema
    const observer = new MutationObserver(applyTheme);
    observer.observe(document.documentElement, { attributes: true, childList: true, subtree: true });
    
    // Aplicar tema inicial
    setTimeout(applyTheme, 100);
    </script>
    """, unsafe_allow_html=True)
      # CSS responsivo completo com suporte a tema escuro
    st.markdown("""
    <style>
    :root {
        --mobile-nav-bg-light: #ffffff;
        --mobile-nav-bg-dark: #262730;
        --mobile-nav-border-light: #e0e0e0;
        --mobile-nav-border-dark: #454545;
        --mobile-nav-text-light: #333333;
        --mobile-nav-text-dark: #ffffff;
        --mobile-nav-hover-light: #f8f9fa;
        --mobile-nav-hover-dark: #3d3d3d;
        --mobile-nav-active-light: #e3f2fd;
        --mobile-nav-active-dark: #1e3a5f;
        --mobile-nav-active-text-light: #1976d2;
        --mobile-nav-active-text-dark: #4fc3f7;
    }
    
    /* Forçar ocultação do sidebar em mobile - seletores mais específicos */
    @media (max-width: 768px) {
        /* Seletores para diferentes versões do Streamlit */
        .css-1d391kg, 
        .st-emotion-cache-r90ti5,
        section[data-testid="stSidebar"],
        .stSidebar,
        div[data-testid="stSidebar"] {
            width: 0 !important;
            min-width: 0 !important;
            max-width: 0 !important;
            overflow: hidden !important;
            transform: translateX(-100%) !important;
            opacity: 0 !important;
            visibility: hidden !important;
        }
        
        /* Ajustar conteúdo principal */
        .css-1lcbmhc, 
        .main .block-container,
        .st-emotion-cache-1y4p8pa,
        .main {
            margin-left: 0 !important;
            padding-left: 0 !important;
            width: 100% !important;
        }
    }
    
    /* Estilo base para o botão de menu mobile com ícone de 3 pontos */
    .mobile-nav-toggle {
        position: fixed;
        top: 1rem;
        right: 1rem;
        background: var(--mobile-nav-bg-light);
        border: 2px solid var(--mobile-nav-border-light);
        border-radius: 12px;
        padding: 14px 16px;
        cursor: pointer;
        z-index: 1001;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        transition: all 0.3s ease;
        display: none;
        min-width: 52px;
        min-height: 52px;
        justify-content: center;
        align-items: center;
    }
    
    [data-theme="dark"] .mobile-nav-toggle {
        background: var(--mobile-nav-bg-dark);
        border-color: var(--mobile-nav-border-dark);
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }    
    .mobile-nav-toggle:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 16px rgba(0,0,0,0.2);
        background: var(--mobile-nav-hover-light);
    }
    
    [data-theme="dark"] .mobile-nav-toggle:hover {
        background: var(--mobile-nav-hover-dark);
        box-shadow: 0 6px 16px rgba(0,0,0,0.4);
    }
    
    .mobile-nav-toggle:active {
        transform: scale(0.95);
    }
    
    /* Ícone de 3 pontos verticais melhorado */
    .mobile-nav-icon {
        font-size: 1.8rem;
        font-weight: 900;
        color: var(--mobile-nav-text-light);
        line-height: 1;
        user-select: none;
        text-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    
    [data-theme="dark"] .mobile-nav-icon {
        color: var(--mobile-nav-text-dark);
        text-shadow: 0 1px 2px rgba(255,255,255,0.1);
    }
    
    /* Menu dropdown mobile */
    .mobile-nav-menu {
        position: fixed;
        top: 4.5rem;
        right: 1rem;
        background: var(--mobile-nav-bg-light);
        border: 1px solid var(--mobile-nav-border-light);
        border-radius: 12px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.15);
        z-index: 1000;
        min-width: 220px;
        max-width: 280px;
        display: none;
        overflow: hidden;
        backdrop-filter: blur(10px);
    }
    
    [data-theme="dark"] .mobile-nav-menu {
        background: var(--mobile-nav-bg-dark);
        border-color: var(--mobile-nav-border-dark);
        box-shadow: 0 8px 24px rgba(0,0,0,0.4);
    }
    
    .mobile-nav-item {
        padding: 16px 20px;
        cursor: pointer;
        transition: all 0.2s ease;
        border-bottom: 1px solid var(--mobile-nav-border-light);
        color: var(--mobile-nav-text-light);
        font-weight: 500;
        font-size: 0.95rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    [data-theme="dark"] .mobile-nav-item {
        border-bottom-color: var(--mobile-nav-border-dark);
        color: var(--mobile-nav-text-dark);
    }
    
    .mobile-nav-item:last-child {
        border-bottom: none;
    }
    
    .mobile-nav-item:hover {
        background-color: var(--mobile-nav-hover-light);
        transform: translateX(4px);
    }
    
    [data-theme="dark"] .mobile-nav-item:hover {
        background-color: var(--mobile-nav-hover-dark);
    }
    
    .mobile-nav-item.active {
        background-color: var(--mobile-nav-active-light);
        color: var(--mobile-nav-active-text-light);
        font-weight: 600;
        border-left: 4px solid var(--mobile-nav-active-text-light);
    }
    
    [data-theme="dark"] .mobile-nav-item.active {
        background-color: var(--mobile-nav-active-dark);
        color: var(--mobile-nav-active-text-dark);
        border-left-color: var(--mobile-nav-active-text-dark);
    }
    
    /* Animações suaves */
    .mobile-nav-menu {
        animation: slideDownFade 0.3s ease-out;
        transform-origin: top right;
    }
    
    @keyframes slideDownFade {
        from {
            opacity: 0;
            transform: translateY(-10px) scale(0.95);
        }
        to {
            opacity: 1;
            transform: translateY(0) scale(1);
        }
    }    
    /* Responsividade */
    @media (max-width: 768px) {
        /* Esconder sidebar padrão do Streamlit - seletores mais abrangentes */
        .css-1d391kg, 
        .st-emotion-cache-r90ti5,
        section[data-testid="stSidebar"],
        .stSidebar,
        div[data-testid="stSidebar"],
        .css-1lcbmhc .css-1d391kg,
        .block-container .css-1d391kg {
            width: 0 !important;
            min-width: 0 !important;
            max-width: 0 !important;
            overflow: hidden !important;
            transform: translateX(-100%) !important;
            opacity: 0 !important;
            visibility: hidden !important;
            display: none !important;
        }
        
        /* Ajustar conteúdo principal */
        .css-1lcbmhc, 
        .main .block-container,
        .st-emotion-cache-1y4p8pa,
        .main,
        .block-container {
            margin-left: 0 !important;
            padding-left: 0 !important;
            width: 100% !important;
            max-width: 100% !important;
        }
        
        /* Mostrar botão mobile */
        .mobile-nav-toggle {
            display: flex !important;
        }
        
        /* Padding para o conteúdo principal não ficar atrás do botão */
        .main > div {
            padding-top: 1rem !important;
        }
    }
    
    @media (min-width: 769px) {
        .mobile-nav-toggle, .mobile-nav-menu {
            display: none !important;
        }
    }
    
    /* Melhoria do sidebar desktop */
    .nav-title {
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: var(--mobile-nav-text-light);
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    [data-theme="dark"] .nav-title {
        color: var(--mobile-nav-text-dark);
    }
    
    /* Estilo melhorado para selectbox */
    .stSelectbox > div > div {
        border-radius: 8px;
        transition: all 0.2s ease;
    }
    
    .stSelectbox > div > div:hover {
        border-color: #4fc3f7;
        box-shadow: 0 0 0 0.2rem rgba(79, 195, 247, 0.25);
    }
    
    /* Ajustes para telas muito pequenas */
    @media (max-width: 480px) {
        .mobile-nav-menu {
            right: 0.5rem;
            left: 0.5rem;
            min-width: auto;
            max-width: none;
        }
        
        .mobile-nav-toggle {
            right: 0.5rem;
        }
    }
    
    /* Indicador visual de página atual */
    .mobile-page-indicator {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        z-index: 998;
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        padding: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        display: none;
    }
    
    [data-theme="dark"] .mobile-page-indicator {
        background: rgba(38, 39, 48, 0.95);
    }
    
    @media (max-width: 768px) {
        .mobile-page-indicator {
            display: block !important;
        }
        
        .main > div {
            padding-top: 4rem !important;
        }
    }
    
    .page-indicator-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 0.9rem;
        font-weight: 500;
        text-align: center;
        margin: 0.5rem 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    }
    </style>
    """, unsafe_allow_html=True)      # JavaScript para controlar a navegação mobile
    st.markdown("""
    <script>
    function isMobile() {
        return window.innerWidth <= 768;
    }
    
    function toggleMobileMenu() {
        const menu = document.querySelector('.mobile-nav-menu');
        if (!menu) return;
        
        const isVisible = menu.style.display === 'block';
        menu.style.display = isVisible ? 'none' : 'block';
        
        // Adicionar efeito de animação
        if (!isVisible) {
            menu.style.animation = 'slideDownFade 0.3s ease-out';
        }
    }
    
    function selectPage(pageName) {
        // Encontrar e atualizar o selectbox
        const selectbox = document.querySelector('[data-testid="stSelectbox"] select');
        if (selectbox) {
            selectbox.value = pageName;
            selectbox.dispatchEvent(new Event('change', { bubbles: true }));
        }
        
        // Fechar menu
        const menu = document.querySelector('.mobile-nav-menu');
        if (menu) menu.style.display = 'none';
        
        // Atualizar estado no Streamlit
        window.parent.postMessage({
            type: 'streamlit:setComponentValue',
            value: pageName
        }, '*');
    }
    
    // Função para forçar ocultação do sidebar em mobile
    function forceSidebarHide() {
        if (isMobile()) {
            const sidebarSelectors = [
                '.css-1d391kg',
                '.st-emotion-cache-r90ti5',
                'section[data-testid="stSidebar"]',
                '.stSidebar',
                'div[data-testid="stSidebar"]'
            ];
            
            sidebarSelectors.forEach(selector => {
                const elements = document.querySelectorAll(selector);
                elements.forEach(element => {
                    element.style.cssText = `
                        width: 0 !important;
                        min-width: 0 !important;
                        max-width: 0 !important;
                        overflow: hidden !important;
                        transform: translateX(-100%) !important;
                        opacity: 0 !important;
                        visibility: hidden !important;
                        display: none !important;
                    `;
                });
            });
            
            // Ajustar conteúdo principal
            const contentSelectors = [
                '.css-1lcbmhc',
                '.main .block-container',
                '.st-emotion-cache-1y4p8pa',
                '.main',
                '.block-container'
            ];
            
            contentSelectors.forEach(selector => {
                const elements = document.querySelectorAll(selector);
                elements.forEach(element => {
                    element.style.cssText += `
                        margin-left: 0 !important;
                        padding-left: 0 !important;
                        width: 100% !important;
                        max-width: 100% !important;
                    `;
                });
            });
        }
    }
    
    // Controlar visibilidade baseada no tamanho da tela
    function handleResize() {
        const toggle = document.querySelector('.mobile-nav-toggle');
        const menu = document.querySelector('.mobile-nav-menu');
        
        if (isMobile()) {
            if (toggle) toggle.style.display = 'flex';
            forceSidebarHide();
        } else {
            if (toggle) toggle.style.display = 'none';
            if (menu) menu.style.display = 'none';
        }
    }
    
    // Fechar menu ao clicar fora
    function handleClickOutside(event) {
        const toggle = document.querySelector('.mobile-nav-toggle');
        const menu = document.querySelector('.mobile-nav-menu');
        
        if (toggle && menu && 
            !toggle.contains(event.target) && 
            !menu.contains(event.target)) {
            menu.style.display = 'none';
        }
    }
    
    // Event listeners
    window.addEventListener('resize', handleResize);
    window.addEventListener('load', handleResize);
    document.addEventListener('click', handleClickOutside);
    
    // Verificação periódica para garantir que os elementos sejam encontrados
    function ensureElements() {
        handleResize();
        forceSidebarHide();
        setTimeout(ensureElements, 500);
    }
    ensureElements();
    
    // Observer para mudanças no DOM
    const observer = new MutationObserver(function(mutations) {
        if (isMobile()) {
            forceSidebarHide();
        }
    });
    
    observer.observe(document.body, { 
        childList: true, 
        subtree: true,
        attributes: true
    });
    </script>
    """, unsafe_allow_html=True)
    
    # Navegação principal (sidebar desktop)
    with st.sidebar:
        st.markdown('<div class="nav-title">🧭 Navegação</div>', unsafe_allow_html=True)
        
        page_options = [
            "🏠 Home",
            "📊 Performance PCA", 
            "⚖️ Comparação MT5",
            "💰 Gestão por Setor",
            "🔬 PCA Avançado", 
            "🔄 Pair Trading"
        ]
        
        page = st.selectbox(
            "Escolha a seção:",
            page_options,
            key="main_nav"
        )
        
        st.markdown("---")    
    # Botão de menu mobile com ícone de 3 pontos melhorado
    st.markdown("""
    <div class="mobile-nav-toggle" onclick="toggleMobileMenu()" title="Menu de navegação">
        <div class="mobile-nav-icon">⋮</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Menu dropdown mobile
    current_page = st.session_state.get('main_nav', '🏠 Home')
    mobile_menu_items = ""
    for option in page_options:
        active_class = "active" if option == current_page else ""
        mobile_menu_items += f'''
            <div class="mobile-nav-item {active_class}" onclick="selectPage('{option}')">
                {option}
            </div>
        '''
    
    st.markdown(f"""
    <div class="mobile-nav-menu">
        {mobile_menu_items}
    </div>
    """, unsafe_allow_html=True)
    
    # Indicador visual da página atual no mobile
    if 'main_nav' in st.session_state:
        current_page_title = st.session_state['main_nav']
        st.markdown(f"""
        <div class="mobile-page-indicator">
            <div class="page-indicator-badge">
                📍 {current_page_title}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
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
