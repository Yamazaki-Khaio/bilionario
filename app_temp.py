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
