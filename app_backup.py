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
import streamlit as st

warnings.filterwarnings("ignore")

# Adicionar import do parser MT5
from mt5_parser import MT5ReportParser

# Importar novos módulos
from pca_advanced import PCAAdvancedAnalysis
from pair_trading import PairTradingAnalysis
from portfolio_allocation import PortfolioAllocationManager
from data_fetch import ASSET_CATEGORIES

# Constantes para literais duplicados
MT5_REAL_LABEL = 'MT5 Real'
PCA_PORTFOLIO_LABEL = 'PCA Portfolio'
SYMBOL_COLUMN = 'Símbolo'
PL_ABS_COLUMN = 'P&L_Abs'
LOSS_LABEL = 'Prejuízo'
PROFIT_LABEL = 'Lucro'
TOTAL_RETURN_LABEL = "Retorno Total"
MAX_DRAWDOWN_LABEL = "Max Drawdown"

# Configurar página do Streamlit
st.set_page_config(
    page_title="💰 Análise Bilionário",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================================
# SISTEMA DE NAVEGAÇÃO POR SIDEBAR
# =====================================================================

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

# Sistema de Navegação por Sidebar
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

# Configurar gerador de números aleatórios do numpy
rng = np.random.default_rng(42)

# Adicionar as funções que estão faltando
def calculate_metrics(returns, initial_capital):
    """Calcula métricas de performance do portfolio"""
    # Verificar se há dados suficientes
    if len(returns) == 0:
        return {
            'total_return': 0.0,
            'annual_return': 0.0,
            'annual_volatility': 0.0,
            'max_drawdown': 0.0
        }
    
    # Calcular equity curve
    equity_curve = (1 + returns).cumprod() * initial_capital
    
    # Retorno total
    total_return = (equity_curve.iloc[-1] / initial_capital) - 1
      # Retorno anualizado (assumindo 252 dias úteis)
    n_periods = len(returns)
    if n_periods > 0:
        annual_return = (1 + total_return) ** (252 / n_periods) - 1
    else:
        annual_return = 0.0
    
    # Volatilidade anualizada
    annual_volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0.0
    
    # Drawdown
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
          # Usar numpy corretamente
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
          # Simular drawdown MT5 variável
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

# Adicionar as funções que estavam faltando:
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
            normalize_metric(risk_metrics['mt5']['recovery_factor'], 0, 10),  # Recuperação        normalize_metric(risk_metrics['mt5']['win_rate'], 0, 1) * 10  # Win Rate como consistência
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
# FUNÇÕES DAS PÁGINAS
# =====================================================================

def show_home_page():
    """Página inicial com carregamento de dados"""
    # Logo
    try:
        logo = Image.open('logo.png')
        st.image(logo, width=150)
    except FileNotFoundError:
        st.write("🚀 **Análise de Portfólio - by Khaio Geovan**")

    st.title("🚀 Bilionário - Análise de Portfolio com PCA")
    st.markdown('Bem-vindo! Ferramenta de análise de carteiras.')
    st.markdown('[Acesse online ▶️](https://bilionario-3w62sdcxhsf3i8yfqywoaq.streamlit.app/)')

    # Botão para baixar dados
    st.markdown("---")
    if st.button('📥 Baixar dados dos ativos'):
        from data_fetch import fetch_data
        fetch_data()
        st.success('✅ Dados baixados!')

    # Verificar se dados existem
    if os.path.exists(RAW_DATA):
        df = pd.read_csv(RAW_DATA, index_col=0, parse_dates=True)
        st.success(f"📊 Dados carregados: {len(df.columns)} ativos, {len(df)} dias")
        
        # Preview dos dados
        st.subheader("📋 Preview dos Dados")
        st.dataframe(df.tail(10), use_container_width=True)
        
        # Estatísticas básicas
        st.subheader("📊 Estatísticas dos Ativos")
        returns = df.pct_change().dropna()
        stats = pd.DataFrame({
            'Retorno Médio (%)': returns.mean() * 100,
            'Volatilidade (%)': returns.std() * 100,
            'Sharpe (aprox)': returns.mean() / returns.std()
        }).round(3)
        st.dataframe(stats, use_container_width=True)
    else:
        st.warning("⚠️ Dados não encontrados. Clique em 'Baixar dados dos ativos' para começar.")

def show_pca_performance_page():
    """Página de análise de performance PCA"""
    st.title("📊 Performance PCA")
    
    # Verificar se dados existem
    if not os.path.exists(RAW_DATA):
        st.error("❌ Dados não encontrados. Vá para a página Home e baixe os dados primeiro.")
        return
    
    df = pd.read_csv(RAW_DATA, index_col=0, parse_dates=True)
    returns = df.pct_change().dropna()
    
    # Seleção de ativos
    if 'selected' not in st.session_state:
        st.session_state['selected'] = df.columns.tolist()[:5]
    
    st.sidebar.subheader("🎯 Seleção de Ativos")
    
    if st.sidebar.button('🎲 Auto-seleção'):
        monthly = get_monthly_returns(returns)
        best, combo = -1, None
        for k in range(3, min(10,len(df.columns))+1):
            for c in itertools.combinations(df.columns,k):
                avg = monthly[list(c)].mean(axis=1).mean()
                if avg>best: 
                    best, combo = avg, c
        st.session_state['selected'] = list(combo)
        st.sidebar.success(f"✅ Selecionados: {len(combo)} ativos")
        
    selected = st.sidebar.multiselect(
        'Selecione ativos (3-20)', 
        df.columns.tolist(),
        default=st.session_state['selected']
    )
    
    if not 3<=len(selected)<=20:
        st.warning('⚠️ Selecione 3 a 20 ativos para análise PCA')
        return
        
    df = df[selected]
    returns = returns[selected]
    
    # Análise PCA
    st.subheader("🔢 Performance PCA")
    initial_capital = st.number_input('💰 Capital Inicial (R$)',100.0,1e7,10000.0,100.0)
    
    if returns.empty or len(returns) == 0:
        st.warning("⚠️ Não há dados suficientes para calcular as métricas.")
        return
    
    portf_ret = returns.mean(axis=1)
    
    if len(portf_ret) == 0:
        st.warning("❌ Erro no cálculo dos retornos do portfólio.")
        return
    
    portf_cum = (1+portf_ret).cumprod()*initial_capital
    metrics = calculate_metrics(portf_ret, initial_capital)
    
    # Métricas principais
    c1,c2,c3,c4 = st.columns(4)
    c1.metric(TOTAL_RETURN_LABEL,f"{metrics['total_return']:.2%}")
    c2.metric("Retorno Anualizado",f"{metrics['annual_return']:.2%}")
    c3.metric("Volatilidade",f"{metrics['annual_volatility']:.2%}")
    c4.metric(MAX_DRAWDOWN_LABEL,f"{metrics['max_drawdown']:.2%}")
    
    # Gráficos
    st.line_chart(portf_cum, height=250)
    
    fig = px.scatter(x=[metrics['annual_volatility']],y=[metrics['annual_return']],
                     text=['PCA'],labels={'x':'Volatilidade','y':'Retorno'},title='Risco vs Retorno')
    fig.update_traces(textposition='top center')
    st.plotly_chart(fig, use_container_width=True)

    # Retorno Mensal
    st.subheader("📅 Retorno Mensal PCA")
    mon = get_monthly_returns(portf_ret)
    st.bar_chart(mon)

    # PCA Analysis
    st.subheader("🔍 Análise PCA")
    n_feats = returns.shape[1]
    n_comp = st.sidebar.slider('Componentes PCA',1,n_feats,min(5,n_feats))
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(returns)
    pca = PCA(n_components=n_comp, random_state=42)
    components = pca.fit_transform(scaled_data)
    
    # Scree plot
    explained_var = pca.explained_variance_ratio_
    fig_scree = px.bar(x=range(1,len(explained_var)+1), y=explained_var*100,
                       labels={'x':'Componente','y':'Variância Explicada (%)'},
                       title='Scree Plot - Variância Explicada')
    st.plotly_chart(fig_scree, use_container_width=True)
    
    # Scatter dos primeiros 2 componentes
    if n_comp >= 2:
        fig_scatter = px.scatter(x=components[:,0], y=components[:,1],
                                title='Primeiros 2 Componentes Principais')
        st.plotly_chart(fig_scatter, use_container_width=True)

def load_mt5_data():
    """Carrega dados MT5 se disponível"""
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

# Verificar se myfxbook está disponível
try:
    from myfxbook_parser import MyfxbookParser
    MYFXBOOK_AVAILABLE = True
except ImportError:
    MYFXBOOK_AVAILABLE = False

# =====================================================================
# CONTROLE PRINCIPAL DE NAVEGAÇÃO  
# =====================================================================

def main():
    """Função principal que controla a navegação"""
    
    # Carregar dados MT5 no sidebar
    st.sidebar.markdown("---")
    mt5_data = load_mt5_data()
    
    # Navegação baseada na página selecionada
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
    st.success("✅ Myfxbook carregado com sucesso!")
    
    # Mostrar apenas informações importantes, não o JSON completo
    if myfx_data:
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Conta:** {myfx_data.get('account_name', 'N/A')}")
            st.write(f"**Ganho:** {myfx_data.get('gain_percent', 'N/A')}")
            st.write(f"**Drawdown:** {myfx_data.get('drawdown_percent', 'N/A')}")
        with col2:
            st.write(f"**Total Trades:** {myfx_data.get('total_trades', 'N/A')}")
            st.write(f"**Pips:** {myfx_data.get('total_pips', 'N/A')}")
            st.write(f"**Win Rate:** {myfx_data.get('win_rate', 'N/A')}")
    
    # Remover: st.json(myfx_data)

# --- Baixar dados ---
st.markdown("---")
if st.button('Baixar dados dos ativos'):
    from data_fetch import fetch_data
    fetch_data()
    st.success('Dados baixados!')

# --- Carregar dados ---
if os.path.exists(RAW_DATA):
    df = pd.read_csv(RAW_DATA, index_col=0, parse_dates=True)
    returns = df.pct_change().dropna()
    
    if 'selected' not in st.session_state:
        st.session_state['selected'] = df.columns.tolist()[:5]
    
    if st.sidebar.button('Auto-seleção'):
        monthly = get_monthly_returns(returns)
        best, combo = -1, None
        for k in range(3, min(10,len(df.columns))+1):
            for c in itertools.combinations(df.columns,k):
                avg = monthly[list(c)].mean(axis=1).mean()
                if avg>best: 
                    best, combo = avg, c
        st.session_state['selected'] = list(combo)
        st.sidebar.success(f"{combo}")
        
    selected = st.sidebar.multiselect('Selecione (3-20)', df.columns.tolist(),
                                      default=st.session_state['selected'])
    if not 3<=len(selected)<=20:
        st.warning('Selecione 3 a 20 ativos para análise PCA')
        st.stop()
        
    df = df[selected]
    returns = returns[selected]    # --- Performance PCA ---
    st.markdown("---")
    st.subheader("🔢 Performance PCA")
    initial_capital = st.number_input('Capital Inicial (R$)',100.0,1e7,10000.0,100.0)
    
    # Verificar se há dados válidos
    if returns.empty or len(returns) == 0:
        st.warning("Não há dados suficientes para calcular as métricas.")
        st.stop()
    
    portf_ret = returns.mean(axis=1)
    
    # Verificar se portf_ret não está vazio
    if len(portf_ret) == 0:
        st.warning("Erro no cálculo dos retornos do portfólio.")
        st.stop()
    
    portf_cum = (1+portf_ret).cumprod()*initial_capital
    metrics = calculate_metrics(portf_ret, initial_capital)
    
    c1,c2,c3,c4 = st.columns(4)
    c1.metric(TOTAL_RETURN_LABEL,f"{metrics['total_return']:.2%}")
    c2.metric("Retorno Anualizado",f"{metrics['annual_return']:.2%}")
    c3.metric("Volatilidade",f"{metrics['annual_volatility']:.2%}")
    c4.metric(MAX_DRAWDOWN_LABEL,f"{metrics['max_drawdown']:.2%}")
    
    st.line_chart(portf_cum, height=250)
    fig = px.scatter(x=[metrics['annual_volatility']],y=[metrics['annual_return']],
                     text=['PCA'],labels={'x':'Volatilidade','y':'Retorno'},title='Risco vs Retorno')
    fig.update_traces(textposition='top center')
    st.plotly_chart(fig)

    # --- Comparação Normalizada PCA vs MT5 ---
    if mt5_data:
        st.markdown("---")
        st.subheader("🔄 Comparação Normalizada")
        normalize = st.checkbox("Normalizar com base no capital inicial", value=True)

        mt5_profit = mt5_data.get('net_profit', 0)
        mt5_balance = mt5_data.get('balance', 0)
        mt5_init_real = mt5_data.get('initial_capital', mt5_balance - mt5_profit)
        pca_final_value = portf_cum.iloc[-1]

        common_capital = min(initial_capital, mt5_init_real) if normalize else 1
        
        pca_return = (pca_final_value / initial_capital) - 1
        mt5_return = mt5_profit / mt5_init_real if mt5_init_real > 0 else 0

        pca_equity = common_capital * (1 + pca_return)
        mt5_equity = common_capital * (1 + mt5_return)

        colA, colB = st.columns(2)
        with colA:
            st.markdown("**PCA Portfolio**")
            st.metric(TOTAL_RETURN_LABEL, f"{pca_return:.2%}")
            st.metric("Equity Final", f"R$ {pca_equity:,.2f}")
        with colB:
            st.markdown("**MT5 Report**")
            st.metric(TOTAL_RETURN_LABEL, f"{mt5_return:.2%}")
            st.metric("Equity Final", f"R$ {mt5_equity:,.2f}")

        comp_df = pd.DataFrame({
            'PCA': [pca_equity],
            'MT5': [mt5_equity]
        }, index=['Equity Normalizado'] if normalize else ['Equity Absoluto'])
        st.bar_chart(comp_df.T)

    # --- Retorno Mensal ---
    st.markdown("---")
    st.subheader("📅 Retorno Mensal PCA")
    mon = get_monthly_returns(portf_ret)
    st.bar_chart(mon)

    # --- PCA Scree & Scatter ---
    st.markdown("---")
    st.subheader("🔍 PCA Analysis")
    n_feats = returns.shape[1]
    n_comp = st.sidebar.slider('Componentes PCA',1,n_feats,min(5,n_feats))
    pca = PCA(n_components=n_comp,random_state=42)
    comps = pca.fit_transform(returns)
    var = pca.explained_variance_ratio_
    
    fig1, ax1 = plt.subplots()
    ax1.bar(range(1,n_comp+1),var)
    ax1.set_ylabel('Var Explicada')
    st.pyplot(fig1)
    
    if n_comp>=2:
        fig2, ax2 = plt.subplots()
        ax2.scatter(comps[:,0],comps[:,1],alpha=0.5)
        ax2.set_xlabel('PC1')
        ax2.set_ylabel('PC2')
        ax2.set_title('PCA Scatter Plot')
        st.pyplot(fig2)
    
    # --- Export StrategyQuant ---
    st.markdown("---")
    st.subheader("💾 Export StrategyQuant")
    if st.button('Exportar'):
        from export_strategyquant import export_to_excel
        export_to_excel()
        st.success("Gerado strategyquant_data.xlsx")
        for fn,m in [('strategyquant_data.xlsx','application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'),
                     ('strategyquant_params.csv','text/csv'),
                     ('strategyquant_trade_config.csv','text/csv')]:
            try:
                with open(fn,'rb') as f:
                    st.download_button(f"Baixar {fn}",f.read(),file_name=fn,mime=m)
            except FileNotFoundError:
                st.warning(f"{fn} não encontrado")

    # --- Comparação Avançada PCA vs MT5 ---
    if mt5_data:
        st.markdown("---")
        st.subheader("📊 Análise Comparativa Avançada PCA vs MT5")
        
        # Tabs para organizar melhor
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Performance", "🎯 Análise Multidimensional", 
            "📊 Alocação MT5", "👤 Perfil Investidor"
        ])
        
        with tab1:
            st.markdown("### 📈 Comparação de Performance")
              # Configurações
            col1, col2 = st.columns(2)
            with col1:
                normalize_adv = st.checkbox("Normalizar capital inicial", value=True, key="normalize_advanced")
            with col2:
                show_detailed_metrics = st.checkbox("Mostrar métricas detalhadas", value=True, key="show_detailed_metrics_main")

            # Extração segura de dados MT5
            mt5_profit = mt5_data.get('net_profit', 0)
            mt5_balance = mt5_data.get('balance', 0)
            mt5_initial = mt5_data.get('initial_capital', mt5_balance - mt5_profit if mt5_balance > 0 else 10000)
            
            # Dados PCA normalizados
            pca_final_value = portf_cum.iloc[-1]
            common_capital = min(initial_capital, mt5_initial) if normalize_adv else initial_capital
            
            pca_return = (pca_final_value / initial_capital) - 1
            mt5_return = mt5_profit / mt5_initial if mt5_initial > 0 else 0
            
            # Equity normalizado
            pca_equity_norm = common_capital * (1 + pca_return)
            mt5_equity_norm = common_capital * (1 + mt5_return)
            
            # Dashboard principal
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(f"**PCA - {TOTAL_RETURN_LABEL}**", 
                         f"{pca_return:.2%}",
                         delta=f"{(pca_return - mt5_return)*100:.2f}pp")
            
            with col2:
                st.metric(f"**MT5 - {TOTAL_RETURN_LABEL}**", 
                         f"{mt5_return:.2%}",
                         delta=f"{(mt5_return - pca_return)*100:.2f}pp")
            
            with col3:
                st.metric("**PCA - Equity Final**", 
                         f"R$ {pca_equity_norm:,.2f}")
            
            with col4:
                st.metric("**MT5 - Equity Final**", 
                         f"R$ {mt5_equity_norm:,.2f}")

            # Gráfico comparativo
            comp_df = pd.DataFrame({
                'Estratégia': [PCA_PORTFOLIO_LABEL, MT5_REAL_LABEL],
                'Equity Final (R$)': [pca_equity_norm, mt5_equity_norm],
                'Retorno (%)': [pca_return * 100, mt5_return * 100],                'Performance': ['PCA', 'MT5']
            })
            
            fig_comp = px.bar(comp_df, x='Estratégia', y='Equity Final (R$)',
                 color='Performance', 
                             title='Comparação de Equity Normalizado',
                             color_discrete_map={'PCA': 'blue', 'MT5': 'red'})
            st.plotly_chart(fig_comp, use_container_width=True, key="comp_chart_tab1")
            
            # Evolução Temporal
            st.markdown("### 📈 Evolução Temporal")
            temporal_fig = plot_temporal_comparison(portf_ret, mt5_data, common_capital)
            if temporal_fig:
                st.plotly_chart(temporal_fig, use_container_width=True, key="temporal_chart_tab1")
        
        with tab2:
            st.markdown("### 📈 Comparação de Performance")
              # Configurações
            col1, col2 = st.columns(2)
            with col1:
                normalize_adv = st.checkbox("Normalizar capital inicial", value=True, key="normalize_advanced_mt5")
            with col2:
                show_detailed_metrics = st.checkbox("Mostrar métricas detalhadas", value=True, key="show_detailed_metrics_tab2")

            # Extração segura de dados MT5
            mt5_profit = mt5_data.get('net_profit', 0)
            mt5_balance = mt5_data.get('balance', 0)
            mt5_initial = mt5_data.get('initial_capital', mt5_balance - mt5_profit if mt5_balance > 0 else 10000)
            mt5_equity = mt5_data.get('equity', mt5_balance)
            
            # Dados PCA normalizados
            pca_final_value = portf_cum.iloc[-1]
            common_capital = min(initial_capital, mt5_initial) if normalize_adv else initial_capital
            
            pca_return = (pca_final_value / initial_capital) - 1
            mt5_return = mt5_profit / mt5_initial if mt5_initial > 0 else 0
            
            # Equity normalizado
            pca_equity_norm = common_capital * (1 + pca_return)
            mt5_equity_norm = common_capital * (1 + mt5_return)
              # Dashboard principal
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(f"**PCA - {TOTAL_RETURN_LABEL}**", 
                         f"{pca_return:.2%}",
                         delta=f"{(pca_return - mt5_return)*100:.2f}pp")
            
            with col2:
                st.metric(f"**MT5 - {TOTAL_RETURN_LABEL}**", 
                         f"{mt5_return:.2%}",                         delta=f"{(mt5_return - pca_return)*100:.2f}pp")
            
            with col3:
                st.metric("**PCA - Equity Final**", 
                         f"R$ {pca_equity_norm:,.2f}")
            
            with col4:
                st.metric("**MT5 - Equity Final**", 
                         f"R$ {mt5_equity_norm:,.2f}")            # Gráfico comparativo
            comp_df = pd.DataFrame({
                'Estratégia': [PCA_PORTFOLIO_LABEL, MT5_REAL_LABEL],
                'Equity Final (R$)': [pca_equity_norm, mt5_equity_norm],
                'Retorno (%)': [pca_return * 100, mt5_return * 100],
                'Performance': ['PCA', 'MT5']
            })
            
            fig_comp = px.bar(comp_df, x='Estratégia', y='Equity Final (R$)',
                             color='Performance', 
                             title='Comparação de Equity Normalizado',
                             color_discrete_map={'PCA': 'blue', 'MT5': 'red'})
            st.plotly_chart(fig_comp, use_container_width=True, key="comp_chart_tab2")

            # Evolução Temporal
            st.markdown("### 📈 Evolução Temporal")
            temporal_fig = plot_temporal_comparison(portf_ret, mt5_data, common_capital)
            if temporal_fig:
                st.plotly_chart(temporal_fig, use_container_width=True, key="temporal_chart_tab2")            # Análise de Drawdown
            st.markdown("### 📉 Análise de Drawdown")
            dd_fig = plot_drawdown_comparison(portf_ret, mt5_data)
            if dd_fig:
                st.plotly_chart(dd_fig, use_container_width=True, key="drawdown_chart_tab2")

            # Métricas detalhadas
            if show_detailed_metrics:
                risk_metrics = create_risk_metrics_analysis(metrics, mt5_data)
                if risk_metrics:
                    st.markdown("### 📊 Métricas de Risco Detalhadas")
                    
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
        
        with tab2:
            st.markdown("### 🎯 Análise Multidimensional")
            
            risk_metrics = create_risk_metrics_analysis(metrics, mt5_data)
            if risk_metrics:
                radar_fig = create_performance_radar_chart(metrics, mt5_data, risk_metrics)
                if radar_fig:
                    st.plotly_chart(radar_fig, use_container_width=True, key="radar_chart_tab2")
                    
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
                    st.plotly_chart(allocation_analysis['pie'], use_container_width=True, key="allocation_pie_tab3")
                with col2:
                    st.plotly_chart(allocation_analysis['bar'], use_container_width=True, key="allocation_bar_tab3")
                
                st.markdown("### 📋 Detalhamento por Símbolo")
                st.dataframe(allocation_analysis['data'], use_container_width=True)

def show_mt5_comparison_page():
    """Página de comparação MT5 vs PCA"""
    st.title("⚖️ Comparação MT5 vs PCA")
    
    # Carregar dados MT5
    mt5_data = load_mt5_data()
    
    if not mt5_data:
        st.warning("⚠️ Faça upload do relatório MT5 para ver a comparação.")
        return
    
    # Verificar se dados PCA existem
    if not os.path.exists(RAW_DATA):
        st.error("❌ Dados PCA não encontrados. Vá para a página Home primeiro.")
        return
        
    df = pd.read_csv(RAW_DATA, index_col=0, parse_dates=True)
    returns = df.pct_change().dropna()
    
    # Usar seleção padrão se não existir
    if 'selected' not in st.session_state:
        st.session_state['selected'] = df.columns.tolist()[:5]
    
    selected = st.session_state['selected']
    
    if len(selected) < 3:
        st.warning("⚠️ Selecione pelo menos 3 ativos na página Performance PCA primeiro.")
        return
        
    df = df[selected]
    returns = returns[selected]
    
    initial_capital = 10000.0
    portf_ret = returns.mean(axis=1)
    metrics = calculate_metrics(portf_ret, initial_capital)
    
    # Comparação lado a lado
    st.subheader("📊 Comparação de Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🤖 PCA Portfolio")
        st.metric("Retorno Total", f"{metrics['total_return']:.2%}")
        st.metric("Retorno Anualizado", f"{metrics['annual_return']:.2%}")
        st.metric("Volatilidade", f"{metrics['annual_volatility']:.2%}")
        st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
        
    with col2:
        st.markdown("### 📈 MT5 Real")
        st.metric("Retorno Total", mt5_data['gain'])
        st.metric("Lucro Líquido", f"R$ {mt5_data['net_profit']:,.2f}")
        st.metric("Saldo", f"R$ {mt5_data['balance']:,.2f}")
        st.metric("Drawdown", mt5_data.get('drawdown', 'N/A'))
        
    # Análise detalhada em tabs
    tab1, tab2, tab3 = st.tabs(["📊 Métricas", "🎯 Análise Radar", "📋 Alocação MT5"])
    
    with tab1:
        st.markdown("### 📊 Métricas de Risco Detalhadas")
        
        # Toggle para métricas detalhadas
        show_detailed_metrics = st.toggle("Mostrar métricas avançadas", value=False)
        
        # Métricas detalhadas
        if show_detailed_metrics:
            risk_metrics = create_risk_metrics_analysis(metrics, mt5_data)
            if risk_metrics:
                st.markdown("### 📊 Métricas de Risco Detalhadas")
                
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
        
        with tab2:
            st.markdown("### 🎯 Análise Multidimensional")
            
            risk_metrics = create_risk_metrics_analysis(metrics, mt5_data)
            if risk_metrics:
                radar_fig = create_performance_radar_chart(metrics, mt5_data, risk_metrics)
                if radar_fig:
                    st.plotly_chart(radar_fig, use_container_width=True, key="radar_chart_tab2")
                    
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
                    st.plotly_chart(allocation_analysis['pie'], use_container_width=True, key="allocation_pie_tab3")
                with col2:
                    st.plotly_chart(allocation_analysis['bar'], use_container_width=True, key="allocation_bar_tab3")
                
                st.markdown("### 📋 Detalhamento por Símbolo")
                st.dataframe(allocation_analysis['data'], use_container_width=True)

def show_sector_management_page():
    """Página de gestão por setor"""
    st.title("💰 Gestão por Setor")
    
    # Verificar se dados existem
    if not os.path.exists(RAW_DATA):
        st.error("❌ Dados não encontrados. Vá para a página Home primeiro.")
        return
        
    df = pd.read_csv(RAW_DATA, index_col=0, parse_dates=True)
    returns = df.pct_change().dropna()
    
    # Seleção de ativos por setor
    st.sidebar.subheader("🏭 Configuração por Setor")
    
    # Definir setores disponíveis
    available_sectors = list(ASSET_CATEGORIES.keys())
    selected_sectors = st.sidebar.multiselect(
        "Setores para análise:",
        available_sectors,
        default=available_sectors[:3]
    )
    
    if not selected_sectors:
        st.warning("⚠️ Selecione pelo menos um setor para análise.")
        return
    
    # Configurar orçamento por setor
    st.subheader("💼 Alocação de Orçamento por Setor")
    
    total_budget = st.number_input("💰 Orçamento Total (R$)", 10000.0, 1e7, 100000.0, 1000.0)
    
    sector_budgets = {}
    cols = st.columns(len(selected_sectors))
    
    for i, sector in enumerate(selected_sectors):
        with cols[i]:
            budget_pct = st.slider(
                f"% {sector}",
                0, 100, 
                100 // len(selected_sectors),
                key=f"budget_{sector}"
            )
            sector_budgets[sector] = budget_pct
    
    # Verificar se soma 100%
    total_pct = sum(sector_budgets.values())
    if total_pct != 100:
        st.warning(f"⚠️ Total deve ser 100%. Atual: {total_pct}%")
        # Normalizar automaticamente
        for sector in sector_budgets:
            sector_budgets[sector] = (sector_budgets[sector] / total_pct) * 100
    
    # Seleção de ativos por setor
    selected = []
    st.subheader("🎯 Seleção de Ativos por Setor")
    
    for sector in selected_sectors:
        st.markdown(f"### 🏭 {sector}")
        available_assets = ASSET_CATEGORIES.get(sector, [])
        available_in_data = [asset for asset in available_assets if asset in df.columns]
        
        if not available_in_data:
            st.warning(f"Nenhum ativo do setor {sector} encontrado nos dados.")
            continue
            
        sector_assets = st.multiselect(
            f"Ativos do setor {sector}:",
            available_in_data,
            default=available_in_data[:2],
            key=f"assets_{sector}"
        )
        selected.extend(sector_assets)
    
    if len(selected) < 3:
        st.warning("⚠️ Selecione pelo menos 3 ativos no total.")
        return
    
    # Análise com gestão por setor
    df = df[selected]
    returns = returns[selected]
    
    # Método de alocação
    allocation_method = st.selectbox(
        "Método de Alocação:",
        ["equal_weight", "pca_weight", "risk_parity", "market_cap"]
    )
    
    if st.button("🚀 Executar Análise por Setor"):
        with st.spinner("Calculando alocação..."):
            # Inicializar gestor de alocação
            allocation_manager = PortfolioAllocationManager(df, returns)
            allocation_manager.set_sector_allocation(sector_budgets)
            
            # Calcular pesos
            portfolio_weights = allocation_manager.calculate_portfolio_weights(
                selected, allocation_method
            )
            
            # Mostrar alocação
            st.subheader("📊 Visualização da Alocação")
            
            # Gráfico de pizza da alocação
            fig_allocation = allocation_manager.plot_sector_allocation()
            if fig_allocation:
                st.plotly_chart(fig_allocation, use_container_width=True, key="sector_allocation_chart")
            
            # Calcular performance por setor
            sector_performance = allocation_manager.calculate_sector_performance(
                selected, portfolio_weights
            )
            
            if sector_performance:
                # Comparação de performance
                fig_comparison = allocation_manager.plot_sector_performance_comparison(sector_performance)
                if fig_comparison:
                    st.plotly_chart(fig_comparison, use_container_width=True, key="sector_performance_chart")
                
                # Tabela de performance
                st.subheader("📊 Performance por Setor")
                performance_df = pd.DataFrame(sector_performance).T
                st.dataframe(performance_df, use_container_width=True)

def show_advanced_pca_page():
    """Página de PCA avançado"""
    st.title("🔬 PCA Avançado")
    
    # Verificar se dados existem
    if not os.path.exists(RAW_DATA):
        st.error("❌ Dados não encontrados. Vá para a página Home primeiro.")
        return
        
    df = pd.read_csv(RAW_DATA, index_col=0, parse_dates=True)
    returns = df.pct_change().dropna()
    
    # Usar seleção de ativos existente ou padrão
    if 'selected' not in st.session_state:
        st.session_state['selected'] = df.columns.tolist()[:8]
    
    selected = st.session_state['selected']
    
    if len(selected) < 3:
        st.warning("⚠️ Selecione ativos na página Performance PCA primeiro.")
        return
        
    df = df[selected]
    returns = returns[selected]
    
    # Inicializar análise PCA avançada
    pca_advanced = PCAAdvancedAnalysis(returns)
    
    # Configurações avançadas
    st.sidebar.subheader("⚙️ Configurações PCA")
    
    # Parâmetros de análise
    rolling_window = st.sidebar.slider("Janela Rolling (dias)", 30, 252, 60)
    min_variance_explained = st.sidebar.slider("Variância Mínima Explicada (%)", 50, 95, 80)
    rebalance_frequency = st.sidebar.selectbox(
        "Frequência de Rebalanceamento:",
        ["daily", "weekly", "monthly", "quarterly"]
    )
    
    # Análises disponíveis
    st.subheader("🔍 Análises Disponíveis")
    
    analysis_tabs = st.tabs([
        "📊 PCA Dinâmico", 
        "🔄 Rolling PCA", 
        "⚖️ Risk Factors", 
        "🎯 Regime Detection",
        "📈 Backtest Avançado"
    ])
    
    with analysis_tabs[0]:
        st.markdown("### 📊 Análise PCA Dinâmica")
        
        if st.button("🚀 Executar PCA Dinâmico"):
            with st.spinner("Calculando PCA dinâmico..."):
                # Análise de componentes principais dinâmica
                results = pca_advanced.dynamic_pca_analysis(
                    window=rolling_window,
                    min_variance=min_variance_explained/100
                )
                
                if results:
                    st.success("✅ Análise concluída!")
                    
                    # Gráfico de componentes ao longo do tempo
                    fig_components = pca_advanced.plot_component_evolution(results)
                    if fig_components:
                        st.plotly_chart(fig_components, use_container_width=True)
                    
                    # Estabilidade dos componentes
                    stability_metrics = pca_advanced.calculate_component_stability(results)
                    if stability_metrics:
                        st.subheader("📊 Estabilidade dos Componentes")
                        st.dataframe(stability_metrics, use_container_width=True)
    
    with analysis_tabs[1]:
        st.markdown("### 🔄 Rolling PCA Analysis")
        
        if st.button("🔄 Executar Rolling PCA"):
            with st.spinner("Calculando rolling PCA..."):
                rolling_results = pca_advanced.rolling_pca_analysis(
                    window=rolling_window,
                    step=5
                )
                
                if rolling_results:
                    st.success("✅ Rolling PCA concluído!")
                    
                    # Variância explicada ao longo do tempo
                    fig_variance = pca_advanced.plot_rolling_variance_explained(rolling_results)
                    if fig_variance:
                        st.plotly_chart(fig_variance, use_container_width=True)
    
    with analysis_tabs[2]:
        st.markdown("### ⚖️ Análise de Fatores de Risco")
        
        if st.button("⚖️ Analisar Fatores de Risco"):
            with st.spinner("Identificando fatores de risco..."):
                risk_factors = pca_advanced.identify_risk_factors()
                
                if risk_factors:
                    st.success("✅ Fatores identificados!")
                    
                    # Heatmap de correlações
                    fig_heatmap = pca_advanced.plot_risk_factor_heatmap(risk_factors)
                    if fig_heatmap:
                        st.plotly_chart(fig_heatmap, use_container_width=True)

def show_pair_trading_page():
    """Página de pair trading"""
    st.title("🔄 Pair Trading")
    
    # Verificar se dados existem
    if not os.path.exists(RAW_DATA):
        st.error("❌ Dados não encontrados. Vá para a página Home primeiro.")
        return
        
    df = pd.read_csv(RAW_DATA, index_col=0, parse_dates=True)
    
    # Inicializar análise de pair trading
    pair_trading = PairTradingAnalysis(df)
    
    # Configurações de pair trading
    st.sidebar.subheader("⚙️ Configurações Pair Trading")
    
    # Parâmetros
    lookback_period = st.sidebar.slider("Período de Lookback (dias)", 30, 252, 60)
    z_threshold = st.sidebar.slider("Z-Score Threshold", 1.0, 3.0, 2.0, 0.1)
    min_correlation = st.sidebar.slider("Correlação Mínima", 0.5, 0.95, 0.8, 0.05)
    
    # Seleção de ativos para pair trading
    available_assets = df.columns.tolist()
    selected_assets = st.multiselect(
        "Selecione ativos para pair trading:",
        available_assets,
        default=available_assets[:6] if len(available_assets) >= 6 else available_assets
    )
    
    if len(selected_assets) < 2:
        st.warning("⚠️ Selecione pelo menos 2 ativos para pair trading.")
        return
    
    # Análises de pair trading
    pair_tabs = st.tabs([
        "🔍 Identificar Pares", 
        "📊 Análise de Spread", 
        "⚡ Sinais de Trading", 
        "📈 Backtest Pares"
    ])
    
    with pair_tabs[0]:
        st.markdown("### 🔍 Identificação de Pares")
        
        if st.button("🔍 Encontrar Melhores Pares"):
            with st.spinner("Identificando pares cointegrados..."):
                pairs = pair_trading.find_cointegrated_pairs(
                    selected_assets, 
                    lookback_period=lookback_period,
                    min_correlation=min_correlation
                )
                
                if pairs:
                    st.success(f"✅ Encontrados {len(pairs)} pares cointegrados!")
                    
                    # Tabela de pares
                    pairs_df = pd.DataFrame(pairs)
                    st.dataframe(pairs_df, use_container_width=True)
                    
                    # Salvar pares no session_state
                    st.session_state['pairs'] = pairs
                else:
                    st.warning("⚠️ Nenhum par cointegrado encontrado com os critérios selecionados.")
    
    with pair_tabs[1]:
        st.markdown("### 📊 Análise de Spread")
        
        if 'pairs' in st.session_state and st.session_state['pairs']:
            selected_pair_idx = st.selectbox(
                "Selecione um par para análise:",
                range(len(st.session_state['pairs'])),
                format_func=lambda x: f"{st.session_state['pairs'][x]['asset1']} / {st.session_state['pairs'][x]['asset2']}"
            )
            
            if st.button("📊 Analisar Spread"):
                pair = st.session_state['pairs'][selected_pair_idx]
                
                with st.spinner("Analisando spread..."):
                    spread_analysis = pair_trading.analyze_spread(
                        pair['asset1'], 
                        pair['asset2'],
                        lookback_period=lookback_period
                    )
                    
                    if spread_analysis:
                        # Gráfico do spread
                        fig_spread = pair_trading.plot_spread_analysis(spread_analysis)
                        if fig_spread:
                            st.plotly_chart(fig_spread, use_container_width=True)
                        
                        # Estatísticas do spread
                        st.subheader("📊 Estatísticas do Spread")
                        stats_cols = st.columns(4)
                        with stats_cols[0]:
                            st.metric("Média", f"{spread_analysis['mean']:.4f}")
                        with stats_cols[1]:
                            st.metric("Desvio Padrão", f"{spread_analysis['std']:.4f}")
                        with stats_cols[2]:
                            st.metric("Z-Score Atual", f"{spread_analysis['current_zscore']:.2f}")
                        with stats_cols[3]:
                            st.metric("P-Value ADF", f"{spread_analysis['adf_pvalue']:.4f}")
        else:
            st.info("ℹ️ Identifique pares primeiro na aba 'Identificar Pares'.")

# =====================================================================
# CONTROLE PRINCIPAL DE NAVEGAÇÃO
# =====================================================================

def main():
    """Função principal que controla a navegação"""
    
    # Carregar dados MT5 no sidebar
    st.sidebar.markdown("---")
    mt5_data = load_mt5_data()
    
    # Navegação baseada na página selecionada
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
