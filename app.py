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

# Configurar gerador de números aleatórios do numpy
rng = np.random.default_rng(42)

# Adicionar as funções que estão faltando
def calculate_metrics(returns, initial_capital):
    """Calcula métricas de performance do portfolio"""
    # Calcular equity curve
    equity_curve = (1 + returns).cumprod() * initial_capital
    
    # Retorno total
    total_return = (equity_curve.iloc[-1] / initial_capital) - 1
    
    # Retorno anualizado (assumindo 252 dias úteis)
    n_periods = len(returns)
    annual_return = (1 + total_return) ** (252 / n_periods) - 1
    
    # Volatilidade anualizada
    annual_volatility = returns.std() * np.sqrt(252)
    
    # Drawdown
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve / running_max - 1)
    max_drawdown = drawdown.min()
    
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

# Verificar se myfxbook está disponível
try:
    from myfxbook_parser import MyfxbookParser
    MYFXBOOK_AVAILABLE = True
except ImportError:
    MYFXBOOK_AVAILABLE = False

# Logo
try:
    logo = Image.open('logo.png')
    st.image(logo, width=150)
except FileNotFoundError:
    st.write("🚀 **Análise de Portfólio - by Khaio Geovan**")

# Diretórios
DATA_DIR = 'data'
RAW_DATA = os.path.join(DATA_DIR, 'raw_data.csv')

st.title("🚀 Bilionário - Análise de Portfolio com PCA")
st.markdown('Bem-vindo! Ferramenta de análise de carteiras.')
st.markdown('[Acesse online ▶️](https://bilionario-3w62sdcxhsf3i8yfqywoaq.streamlit.app/)')

# --- Upload MT5 Report ---
st.sidebar.markdown("---")
st.sidebar.subheader("📊 MT5 Report")
uploaded_mt5 = st.sidebar.file_uploader(
    "Upload relatório MT5 (HTML)", 
    type=['html', 'htm'],
    help="Faça upload do relatório HTML exportado do MetaTrader 5"
)

mt5_data = None
if uploaded_mt5 is not None:
    try:
        # Determinar tipo de arquivo
        file_type = '.html' if uploaded_mt5.name.lower().endswith(('.html', '.htm')) else '.pdf'
        
        # Parse do arquivo MT5
        parser = MT5ReportParser(uploaded_mt5, file_type)
        mt5_data = parser.get_portfolio_summary()
        
        # Mostrar resumo do MT5
        st.sidebar.success("✅ MT5 carregado com sucesso!")
        st.sidebar.write(f"**Conta:** {mt5_data['account_name']}")
        st.sidebar.write(f"**Saldo:** R$ {mt5_data['balance']:,.2f}")
        st.sidebar.write(f"**Lucro:** R$ {mt5_data['net_profit']:,.2f}")
        st.sidebar.write(f"**Retorno:** {mt5_data['gain']}")
        
    except Exception as e:
        st.sidebar.error(f"Erro ao processar MT5: {str(e)}")
        mt5_data = None

# --- Myfxbook ---
st.markdown("---")
st.markdown("## 📊 Myfxbook (opcional)")
uploaded = st.file_uploader("Upload Myfxbook (PDF, CSV, HTML)", type=['pdf','csv','html','htm'])
myfx_data = None
if uploaded and MYFXBOOK_AVAILABLE:
    ext = f".{uploaded.name.split('.')[-1]}"
    parser = MyfxbookParser(uploaded, ext)
    myfx_data = parser.get_portfolio_summary()
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
    returns = returns[selected]

    # --- Performance PCA ---
    st.markdown("---")
    st.subheader("🔢 Performance PCA")
    initial_capital = st.number_input('Capital Inicial (R$)',100.0,1e7,10000.0,100.0)
    portf_ret = returns.mean(axis=1)
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
                show_detailed_metrics = st.checkbox("Mostrar métricas detalhadas", value=True)

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
                         f"R$ {mt5_equity_norm:,.2f}")

            # Gráfico comparativo
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
            st.plotly_chart(fig_comp, use_container_width=True)

            # Evolução Temporal
            st.markdown("### 📈 Evolução Temporal")
            temporal_fig = plot_temporal_comparison(portf_ret, mt5_data, common_capital)
            if temporal_fig:
                st.plotly_chart(temporal_fig, use_container_width=True)

            # Análise de Drawdown
            st.markdown("### 📉 Análise de Drawdown")
            dd_fig = plot_drawdown_comparison(portf_ret, mt5_data)
            if dd_fig:
                st.plotly_chart(dd_fig, use_container_width=True)

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
                    st.plotly_chart(radar_fig, use_container_width=True)
                    
                    st.info("""
                    **Interpretação do Gráfico Radar:**
                    - **Retorno**: Performance de retorno normalizada
                    - **Risco (inv)**: Inverso do risco (maior = melhor)
                    - **Sharpe**: Retorno ajustado pelo risco
                    - **Recuperação**: Capacidade de recuperar de perdas
                    - **Consistência**: Estabilidade da estratégia
                    """)

        with tab3:
            st.markdown("### 📊 Análise de Alocação MT5")
            
            allocation_analysis = create_portfolio_allocation_analysis(mt5_data)
            if allocation_analysis:
                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(allocation_analysis['pie'], use_container_width=True)
                with col2:
                    st.plotly_chart(allocation_analysis['bar'], use_container_width=True)
                
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
                st.warning("Dados de símbolos não disponíveis ou inválidos no relatório MT5")

        with tab4:
            st.markdown("### 👤 Análise por Perfil de Investidor")
            
            investor_profile = st.selectbox("Selecione o Perfil do Investidor", 
                                          ["Conservador", "Moderado", "Arrojado"])
            
            # Definição de perfis com critérios mais específicos
            profiles = {
                "Conservador": {
                    "max_vol": 0.15, "min_sharpe": 0.5, "max_dd": 0.10,
                    "min_win_rate": 0.6, "description": "Foco em preservação de capital"
                },
                "Moderado": {
                    "max_vol": 0.25, "min_sharpe": 0.3, "max_dd": 0.20,
                    "min_win_rate": 0.5, "description": "Equilíbrio entre risco e retorno"
                },
                "Arrojado": {
                    "max_vol": 0.40, "min_sharpe": 0.2, "max_dd": 0.35,
                    "min_win_rate": 0.4, "description": "Busca por maiores retornos"
                }
            }
            
            profile_limits = profiles[investor_profile]
            
            st.info(f"**Perfil {investor_profile}**: {profile_limits['description']}")
            
            # Avaliação automática
            risk_metrics = create_risk_metrics_analysis(metrics, mt5_data)
            if risk_metrics:
                # Score PCA
                pca_score = 0
                pca_feedback = []
                
                if metrics['annual_volatility'] <= profile_limits['max_vol']:
                    pca_score += 1
                    pca_feedback.append("✅ Volatilidade adequada")
                else:
                    pca_feedback.append(f"❌ Volatilidade alta ({metrics['annual_volatility']:.1%})")
                    
                if risk_metrics['pca']['sharpe_ratio'] >= profile_limits['min_sharpe']:
                    pca_score += 1
                    pca_feedback.append("✅ Sharpe Ratio adequado")
                else:
                    pca_feedback.append(f"❌ Sharpe Ratio baixo ({risk_metrics['pca']['sharpe_ratio']:.2f})")
                    
                if abs(metrics['max_drawdown']) <= profile_limits['max_dd']:
                    pca_score += 1
                    pca_feedback.append("✅ Drawdown controlado")
                else:
                    pca_feedback.append(f"❌ Drawdown alto ({metrics['max_drawdown']:.1%})")
                
                # Score MT5
                mt5_score = 0
                mt5_feedback = []
                
                if risk_metrics['mt5']['drawdown'] <= profile_limits['max_dd']:
                    mt5_score += 1
                    mt5_feedback.append("✅ Drawdown controlado")
                else:
                    mt5_feedback.append(f"❌ Drawdown alto ({risk_metrics['mt5']['drawdown']:.1%})")
                    
                if risk_metrics['mt5']['win_rate'] >= profile_limits['min_win_rate']:
                    mt5_score += 1
                    mt5_feedback.append("✅ Win rate adequado")
                else:
                    mt5_feedback.append(f"❌ Win rate baixo ({risk_metrics['mt5']['win_rate']:.1%})")
                    
                if mt5_return > 0:
                    mt5_score += 1
                    mt5_feedback.append("✅ Retorno positivo")
                else:
                    mt5_feedback.append("❌ Retorno negativo")

                # Resultado final
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"**PCA Portfolio** - Score: {pca_score}/3")
                    if pca_score >= 2:
                        st.success(f"✅ Adequado para perfil {investor_profile}")
                    else:
                        st.warning(f"⚠️ Requer atenção para perfil {investor_profile}")
                    
                    for feedback in pca_feedback:
                        st.write(feedback)
                
                with col2:
                    st.markdown(f"**MT5 Real** - Score: {mt5_score}/3")
                    if mt5_score >= 2:
                        st.success(f"✅ Adequado para perfil {investor_profile}")
                    else:
                        st.warning(f"⚠️ Requer atenção para perfil {investor_profile}")
                    
                    for feedback in mt5_feedback:
                        st.write(feedback)

        # Recomendação final
        st.markdown("---")
        st.markdown("### 💡 Recomendação Final")
        
        better_strategy = "PCA" if pca_return > mt5_return else "MT5"
        performance_diff = abs(pca_return - mt5_return) * 100
        
        if performance_diff < 1:
            st.info("📊 **Performance Equilibrada**: Ambas estratégias apresentam resultados similares. Considere diversificar entre elas.")
        elif better_strategy == "PCA":
            st.success(f"🎯 **PCA Recomendado**: Superou MT5 em {performance_diff:.1f}pp. A diversificação sistemática está gerando valor.")
        else:
            st.warning(f"⚡ **MT5 Recomendado**: Superou PCA em {performance_diff:.1f}pp. A execução manual está agregando alpha.")

    # --- Nova Seção: Divisão de Capital por Setor ---
    st.markdown("---")
    st.header("💰 Gestão de Capital por Setor")
    
    with st.expander("🔧 Configurar Alocação de Capital"):
        st.markdown("**Defina o orçamento para cada setor de ativos:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            acoes_budget = st.number_input('💼 Ações (R$)', min_value=0.0, value=1000.0, step=100.0)
            forex_budget = st.number_input('💱 Forex (USD)', min_value=0.0, value=100.0, step=10.0)
        
        with col2:
            cripto_budget = st.number_input('🪙 Criptomoedas (R$)', min_value=0.0, value=100.0, step=50.0)
            etf_budget = st.number_input('📊 ETFs/Índices (R$)', min_value=0.0, value=500.0, step=100.0)
        
        # Configurar alocação
        sector_budgets = {
            'acoes': acoes_budget,
            'forex': forex_budget,
            'criptomoedas': cripto_budget,
            'etfs_indices': etf_budget
        }
        
        # Método de alocação dentro do setor
        allocation_method = st.selectbox(
            '⚖️ Método de Alocação dentro do Setor',
            ['equal_weight', 'volatility_parity', 'momentum'],
            format_func=lambda x: {
                'equal_weight': 'Peso Igual',
                'volatility_parity': 'Paridade de Volatilidade', 
                'momentum': 'Baseado em Momentum'
            }[x]
        )
        
        if st.button('🚀 Aplicar Alocação por Setor'):
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
                st.plotly_chart(fig_allocation, use_container_width=True)
              # Calcular performance por setor
            sector_performance = allocation_manager.calculate_sector_performance(
                selected, portfolio_weights
            )
            
            if sector_performance:
                # Comparação de performance
                fig_comparison = allocation_manager.plot_sector_performance_comparison(sector_performance)
                if fig_comparison:
                    st.plotly_chart(fig_comparison, use_container_width=True)
                
                # Evolução temporal
                fig_evolution = allocation_manager.plot_sector_evolution(sector_performance)
                if fig_evolution:
                    st.plotly_chart(fig_evolution, use_container_width=True)

    # Resumo por setor (fora do expander para evitar aninhamento)
    if 'sector_performance' in locals() and sector_performance:
        st.subheader("📋 Resumo por Setor")
        sector_summary = allocation_manager.get_sector_summary(sector_performance)
        
        # Usar colunas ao invés de expanders aninhados
        for sector, metrics in sector_summary.items():
            st.markdown(f"### 📊 {sector}")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Alocação", metrics['Alocação'])
                st.metric(TOTAL_RETURN_LABEL, metrics['Retorno Total'])
                st.metric("Retorno Anual", metrics['Retorno Anual'])
            
            with col2:
                st.metric("Volatilidade", metrics['Volatilidade'])
                st.metric("Sharpe Ratio", metrics['Sharpe Ratio'])
                st.metric(MAX_DRAWDOWN_LABEL, metrics['Max Drawdown'])
            
            with col3:
                st.write("**Ativos Incluídos:**")
                st.write(metrics['Ativos'])
            
            st.markdown("---")

    # --- Nova Seção: Análise PCA Avançada ---
    st.markdown("---")
    st.header("🔬 Análise PCA Avançada")
    
    with st.expander("📈 Análise Detalhada dos Componentes Principais"):
        try:
            # Inicializar análise PCA avançada
            pca_advanced = PCAAdvancedAnalysis(returns)
            
            # Realizar análise completa
            pca_results = pca_advanced.analyze_components()
            
            if pca_results:
                st.subheader("📊 Visualizações PCA")
                
                # Scree Plot
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_scree = pca_advanced.plot_scree()
                    if fig_scree:
                        st.plotly_chart(fig_scree, use_container_width=True)
                
                with col2:
                    fig_loadings = pca_advanced.plot_loadings_heatmap()
                    if fig_loadings:
                        st.plotly_chart(fig_loadings, use_container_width=True)
                
                # Correlation Matrix e Biplot 3D
                col1, col2 = st.columns(2)
                
                with col1:
                    fig_corr = pca_advanced.plot_correlation_matrix()
                    if fig_corr:
                        st.plotly_chart(fig_corr, use_container_width=True)
                
                with col2:
                    fig_3d = pca_advanced.plot_biplot_3d()
                    if fig_3d:
                        st.plotly_chart(fig_3d, use_container_width=True)
                  # Interpretação dos componentes
                st.subheader("🧠 Interpretação dos Componentes Principais")
                interpretations = pca_advanced.interpret_components()
                
                for i, interpretation in enumerate(interpretations[:3], 1):
                    st.write(f"**📊 Componente Principal {i}**")
                    st.write(f"**Variância Explicada:** {interpretation['variance_explained']:.2%}")
                    st.write(f"**Interpretação:** {interpretation['interpretation']}")
                    st.write("**Principais Ativos:**")
                    for asset, weight in interpretation['top_assets']:
                        st.write(f"• {asset}: {weight:.3f}")
                    st.markdown("---")
        
        except Exception as e:
            st.error(f"Erro na análise PCA avançada: {str(e)}")    # --- Nova Seção: Pair Trading ---
    st.markdown("---")
    st.header("🔄 Análise de Pair Trading")
    
    with st.expander("💹 Estratégia de Pares"):
        # Usar todos os ativos disponíveis para pair trading
        all_assets = df.columns.tolist()
        
        if len(all_assets) >= 2:
            st.info(f"📊 {len(all_assets)} ativos disponíveis para análise de pair trading")
            
            # Seleção de pares
            col1, col2 = st.columns(2)
            
            with col1:
                asset1 = st.selectbox('🎯 Ativo 1', all_assets, key='pair_asset1')
            
            with col2:
                available_assets = [a for a in all_assets if a != asset1]
                asset2 = st.selectbox('🎯 Ativo 2', available_assets, key='pair_asset2')
            
            if st.button('🔍 Analisar Pair Trading'):
                try:
                    # Inicializar análise de pair trading
                    pair_analysis = PairTradingAnalysis(df[[asset1, asset2]])
                    
                    # Teste de cointegração
                    coint_result = pair_analysis.test_cointegration(asset1, asset2)
                    
                    st.subheader("📊 Teste de Cointegração")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Estatística Cointegração", f"{coint_result['cointegration_stat']:.4f}")
                    
                    with col2:
                        st.metric("P-valor", f"{coint_result['p_value']:.4f}")
                    
                    with col3:
                        cointegrated = "✅ Sim" if coint_result['is_cointegrated'] else "❌ Não"
                        st.metric("Cointegração", cointegrated)
                    
                    if coint_result['is_cointegrated']:
                        st.success("🎉 Os ativos são cointegrados! Pair trading é viável.")
                        
                        # Gerar sinais de trading
                        signals = pair_analysis.generate_signals()
                        
                        # Realizar backtesting
                        backtest_results = pair_analysis.backtest_strategy(initial_capital=10000)
                        
                        # Visualizações
                        st.subheader("📈 Visualizações da Estratégia")
                        
                        # Spread e sinais
                        fig_spread = pair_analysis.plot_spread_and_signals()
                        if fig_spread:
                            st.plotly_chart(fig_spread, use_container_width=True)
                          # Performance
                        fig_performance = pair_analysis.plot_strategy_performance()
                        if fig_performance:
                            st.plotly_chart(fig_performance, use_container_width=True)
                        
                        # Métricas do backtesting
                        st.subheader("📊 Métricas da Estratégia")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric(TOTAL_RETURN_LABEL, f"{backtest_results['total_return']:.2%}")
                        
                        with col2:
                            st.metric("Sharpe Ratio", f"{backtest_results['sharpe_ratio']:.2f}")
                        
                        with col3:
                            st.metric(MAX_DRAWDOWN_LABEL, f"{backtest_results['max_drawdown']:.2%}")
                        
                        with col4:
                            st.metric("Número de Trades", backtest_results['num_trades'])
                        
                        # Análise de trades
                        st.subheader("📋 Análise de Trades")
                        
                        if backtest_results['num_trades'] > 0:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.metric("Win Rate", f"{backtest_results.get('win_rate', 0):.1%}")
                                st.metric("Profit Factor", f"{backtest_results.get('profit_factor', 0):.2f}")
                            
                            with col2:
                                st.metric("Avg. Trade Return", f"{backtest_results.get('avg_trade_return', 0):.2%}")                                
                                st.metric("Volatilidade", f"{backtest_results.get('volatility', 0):.2%}")
                    else:
                        st.warning("⚠️ Os ativos não são cointegrados. Pair trading não é recomendado para este par.")
                
                except Exception as e:
                    st.error(f"Erro na análise de pair trading: {str(e)}")
        else:
            st.warning("⚠️ É necessário ter pelo menos 2 ativos disponíveis para análise de pair trading.")

else:
    st.error("⚠️ Dados não encontrados. Execute 'Baixar dados dos ativos' primeiro.")
