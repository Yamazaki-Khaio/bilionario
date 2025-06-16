# mt5_comparison_helpers.py
"""Funções auxiliares para reduzir complexidade cognitiva da comparação MT5"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from constants import get_raw_data_path, FINAL_CAPITAL_LABEL, TOTAL_RETURN_LABEL
from financial_formatting import format_currency, format_percentage
from performance_metrics import calculate_metrics


def validate_mt5_data():
    """Valida se há dados MT5 carregados"""
    mt5_data = st.session_state.get('mt5_data')
    
    if not mt5_data:
        st.warning("⚠️ Nenhum dado MT5 carregado. Faça upload de um relatório MT5 no sidebar.")
        st.info("📝 **Como usar:** Carregue um relatório MT5 HTML no sidebar para ativar as comparações avançadas.")
        return None
        
    st.success("✅ Dados MT5 carregados com sucesso!")
    return mt5_data


def validate_pca_data():
    """Valida e carrega dados PCA"""
    RAW_DATA = get_raw_data_path()
    
    if not os.path.exists(RAW_DATA):
        st.error("❌ Dados PCA não encontrados. Vá para a página Home e baixe os dados primeiro.")
        return None, None
    
    df = pd.read_csv(RAW_DATA, index_col=0, parse_dates=True)
    returns = df.pct_change().dropna()
    return df, returns


def setup_mt5_comparison_sidebar():
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


def select_pca_assets(df):
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


def calculate_pca_metrics(returns, selected, initial_capital):
    """Calcula métricas do portfolio PCA"""
    returns_selected = returns[selected]
    portf_ret = returns_selected.mean(axis=1)
    portf_cum = (1 + portf_ret).cumprod() * initial_capital
    pca_metrics = calculate_metrics(portf_ret, initial_capital)
    
    return portf_ret, portf_cum, pca_metrics


def display_pca_summary(pca_metrics, portf_cum, initial_capital):
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


def display_mt5_summary(mt5_data):
    """Exibe resumo do MT5"""
    mt5_balance = mt5_data.get('balance', 0)
    mt5_profit = mt5_data.get('net_profit', 0)
    mt5_initial = mt5_data.get('initial_capital', mt5_balance - mt5_profit)
    mt5_return = mt5_profit / mt5_initial if mt5_initial > 0 else 0
    
    st.metric("Saldo Atual", format_currency(mt5_balance))
    st.metric("Lucro Líquido", format_currency(mt5_profit))
    st.metric(TOTAL_RETURN_LABEL, format_percentage(mt5_return))
    
    return mt5_return


def display_comparative_analysis(pca_metrics, pca_return, mt5_return):
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


def display_recommendations(pca_metrics, mt5_return, pca_return):
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


def plot_comparative_performance(pca_cum, mt5_data, initial_capital):
    """Cria gráfico de desempenho comparativo entre PCA e MT5"""
    try:
        # Criar dataframe comparativo
        mt5_balance = mt5_data.get('balance', initial_capital)
        mt5_initial = mt5_data.get('initial_capital', mt5_balance - mt5_data.get('net_profit', 0))
        
        # Criar série temporal simulada para MT5 (interpolação linear)
        dates = pca_cum.index
        date_range = (dates[-1] - dates[0]).days
        
        # Simulação simples: crescimento linear do capital inicial até o saldo final
        mt5_factor = mt5_balance / mt5_initial
        mt5_daily_growth = mt5_factor ** (1/date_range) if date_range > 0 else 1
        mt5_curve = pd.Series(
            [mt5_initial * (mt5_daily_growth ** (i/(len(dates)-1) * date_range)) for i in range(len(dates))],
            index=dates
        )
        
        # Criar dataframe combinado
        df_combined = pd.DataFrame({
            'PCA Portfolio': pca_cum,
            'MT5 Real': mt5_curve
        })
        
        # Criar gráfico
        fig = px.line(
            df_combined,
            title="Comparação de Performance: PCA vs MT5",
            labels={"value": "Capital (R$)", "variable": "Estratégia"}
        )
        
        fig.update_layout(
            height=500,
            legend_title="Estratégias",
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    except Exception as e:
        st.error(f"Erro ao gerar gráfico comparativo: {str(e)}")


def analyze_symbol_performance(mt5_data):
    """Analisa performance por símbolo no MT5"""
    try:
        symbols = mt5_data.get('symbols', {})
        
        if not symbols:
            st.warning("⚠️ Nenhum dado de símbolo disponível no relatório MT5")
            return
            
        # Preparar dados para visualização
        symbol_data = []
        for symbol, profit in symbols.items():
            symbol_data.append({
                'Símbolo': symbol,
                'Lucro': profit
            })
            
        df_symbols = pd.DataFrame(symbol_data)
        
        if df_symbols.empty:
            st.warning("⚠️ Nenhum símbolo com dados válidos encontrado")
            return
            
        # Ordenar por lucro
        df_symbols = df_symbols.sort_values(by='Lucro', ascending=False)
        
        # Criar visualização        
        fig = px.bar(
            df_symbols,
            x='Símbolo',
            y='Lucro',
            title="Performance por Símbolo no MT5",
            color='Lucro',
            color_continuous_scale=['red', 'green'],
            labels={'Lucro': 'Lucro/Prejuízo (R$)'}
        )
        
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Análise adicional
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Melhores Símbolos")
            best_symbols = df_symbols[df_symbols['Lucro'] > 0].head(5)
            if not best_symbols.empty:
                st.dataframe(best_symbols, use_container_width=True)
            else:
                st.info("Nenhum símbolo com lucro encontrado")
                
        with col2:
            st.subheader("Piores Símbolos")
            worst_symbols = df_symbols[df_symbols['Lucro'] < 0].head(5)
            if not worst_symbols.empty:
                st.dataframe(worst_symbols, use_container_width=True)
            else:
                st.info("Nenhum símbolo com prejuízo encontrado")
        
    except Exception as e:
        st.error(f"Erro na análise por símbolo: {str(e)}")


def create_risk_metrics_radar(pca_metrics, mt5_data):
    """Cria gráfico radar comparativo de métricas de risco"""
    try:
        # Extrair métricas MT5
        mt5_profit = mt5_data.get('net_profit', 0)
        mt5_initial = mt5_data.get('initial_capital', 10000)
        mt5_return = mt5_profit / mt5_initial if mt5_initial > 0 else 0
        mt5_annual_return = ((1 + mt5_return) ** (252/365) - 1) if mt5_return > -1 else -0.99
        mt5_drawdown = float(mt5_data.get('drawdown', '0%').replace('%', '')) / 100
        mt5_profit_factor = mt5_data.get('profit_factor', 1)
        mt5_recovery_factor = mt5_data.get('recovery_factor', 1)
        mt5_win_rate = mt5_data.get('win_rate', 50) / 100
        
        # Métricas PCA
        pca_annual_return = pca_metrics['annual_return']
        pca_volatility = pca_metrics['annual_volatility']
        pca_drawdown = abs(pca_metrics['max_drawdown'])
        pca_sharpe = pca_metrics['annual_return'] / pca_metrics['annual_volatility'] if pca_metrics['annual_volatility'] > 0 else 0
        
        # Normalizar valores para 0-1
        metrics = {
            'Retorno Anualizado': [
                min(max(pca_annual_return / 0.5, 0), 1),  # PCA
                min(max(mt5_annual_return / 0.5, 0), 1)   # MT5
            ],
            'Controle Drawdown': [
                1 - min(pca_drawdown / 0.5, 1),  # PCA
                1 - min(mt5_drawdown / 0.5, 1)   # MT5
            ],
            'Sharpe Ratio': [
                min(max(pca_sharpe / 3, 0), 1),  # PCA
                min(max(mt5_profit_factor / 3, 0), 1)  # MT5
            ],
            'Estabilidade': [
                1 - min(pca_volatility / 0.5, 1),  # PCA
                min(max(mt5_recovery_factor / 3, 0), 1)  # MT5
            ],
            'Consistência': [
                0.7,  # PCA (valor arbitrário)
                min(max(mt5_win_rate, 0), 1)  # MT5
            ]
        }
        
        # Criar categorias e valores para o gráfico
        categories = list(metrics.keys())
        pca_values = [metrics[cat][0] for cat in categories]
        mt5_values = [metrics[cat][1] for cat in categories]
        
        # Fechar o círculo repetindo o primeiro valor
        categories = categories + [categories[0]]
        pca_values = pca_values + [pca_values[0]]
        mt5_values = mt5_values + [mt5_values[0]]
        
        # Criar gráfico
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=pca_values,
            theta=categories,
            fill='toself',
            name='PCA Portfolio',
            line=dict(color='blue')
        ))
        
        fig.add_trace(go.Scatterpolar(
            r=mt5_values,
            theta=categories,
            fill='toself',
            name='MT5 Real',
            line=dict(color='red')
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title="Radar de Performance: PCA vs MT5",
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Erro ao gerar radar de métricas: {str(e)}")
