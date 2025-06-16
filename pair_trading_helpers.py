#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PAIR TRADING HELPERS - FUNÇÕES AUXILIARES
==========================================
Módulo contendo funções auxiliares para reduzir a complexidade cognitiva
da função show_pair_trading_page().
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from constants import (
    TOTAL_RETURN_LABEL, FINAL_CAPITAL_LABEL, RESULTS_INTERPRETATION_LABEL,
    SELECT_VALID_PAIR_MSG
)
from financial_formatting import format_percentage, format_ratio, format_currency
from data_fetch import ASSET_CATEGORIES
from data_fetch import ASSET_CATEGORIES


def setup_pair_trading_sidebar():
    """Configura a sidebar com parâmetros de pair trading"""
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
    
    return {
        'min_correlation': min_correlation,
        'min_data_points': min_data_points,
        'entry_threshold': entry_threshold,
        'exit_threshold': exit_threshold,
        'stop_loss': stop_loss,
        'transaction_cost': transaction_cost
    }


def find_cointegrated_pairs_tab(pair_analyzer, all_assets, params):
    """Aba 1: Identificação de pares cointegrados"""
    st.subheader("🔍 Identificação de Pares Cointegrados")
    
    # Seleção de ativos para análise
    selected_assets = st.multiselect(
        "Selecione ativos para busca de pares (deixe vazio para usar todos):",
        all_assets,
        default=[]
    )
    if not selected_assets:
        selected_assets = all_assets[:20]  # Limitar para performance
        st.info(f"📊 Usando os primeiros 20 ativos: {', '.join(selected_assets[:5])}...")
    
    if st.button("🔍 Buscar Pares Cointegrados"):
        _execute_pair_search(pair_analyzer, selected_assets, params)
    # Matriz de correlação visual
    _display_correlation_matrix(selected_assets, pair_analyzer.price_data)
    
    return selected_assets


def _execute_pair_search(pair_analyzer, selected_assets, params):
    """Executa a busca por pares cointegrados"""
    # selected_assets não é utilizado diretamente, mas mantido por compatibilidade de API
    with st.spinner("Analisando correlações e cointegração..."):
        # Encontrar pares correlacionados
        correlated_pairs = pair_analyzer.find_correlated_pairs(
            min_correlation=params['min_correlation'],
            min_years=params['min_data_points']/252
        )
        
        if correlated_pairs:
            st.success(f"✅ Encontrados {len(correlated_pairs)} pares altamente correlacionados!")
            
            # Testar cointegração
            cointegrated_pairs = _test_cointegration_pairs(
                pair_analyzer, correlated_pairs
            )
            
            if cointegrated_pairs:
                _display_cointegrated_pairs_table(cointegrated_pairs)
                st.session_state['cointegrated_pairs'] = cointegrated_pairs
            else:
                st.warning("⚠️ Nenhum par cointegrado encontrado com os critérios atuais.")
        else:
            st.warning("⚠️ Nenhum par com correlação suficiente encontrado.")


def _test_cointegration_pairs(pair_analyzer, correlated_pairs):
    """Testa cointegração dos pares correlacionados"""
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
    
    return cointegrated_pairs


def _display_cointegrated_pairs_table(cointegrated_pairs):
    """Exibe tabela de pares cointegrados"""
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


def _display_correlation_matrix(selected_assets, df):
    """Exibe matriz de correlação visual"""
    if st.checkbox("📊 Mostrar Matriz de Correlação", key="show_corr_matrix"):
        # Garantir que temos ativos selecionados para exibir
        display_assets = selected_assets
        if not display_assets or len(display_assets) < 2:
            # Se não houver ativos selecionados, usar as primeiras 15 colunas do DataFrame
            display_assets = df.columns.tolist()[:15]
            st.info(f"Usando os primeiros {len(display_assets)} ativos para exibir a matriz de correlação")
            
        if len(display_assets) <= 15:  # Evitar matriz muito grande
            correlation_matrix = df[display_assets].corr()
            
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


def detailed_analysis_tab(pair_analyzer, all_assets, params):
    """Aba 2: Análise detalhada do par"""
    st.subheader("📊 Análise Detalhada do Par")
    # Seleção de ativos melhorada
    st.markdown("### 🎯 Seleção de Ativos")
    
    col1, col2 = st.columns(2)
    with col1:
        # Modo de seleção
        analysis_mode = st.radio(
            "Modo de análise:",
            ["🤖 Usar melhor par encontrado", "✋ Selecionar par manualmente"],
            help="Escolha como selecionar os ativos para análise"
        )
    with col2:
        # Configurações adicionais
        show_correlation = st.checkbox("📊 Mostrar correlação", value=True, key="show_corr_detail")
        show_details = st.checkbox("📋 Mostrar detalhes técnicos", value=False, key="show_tech_details")
    
    # Seleção de ativos baseada no modo escolhido
    # Usar a função _select_assets_for_analysis que estava sendo ignorada
    asset1, asset2 = _select_assets_for_analysis(analysis_mode, all_assets)
    
    # Se temos pares cointegrados nos resultados anteriores, mostrar informações
    if analysis_mode == "🤖 Usar melhor par encontrado":
        if 'cointegrated_pairs' in st.session_state and st.session_state['cointegrated_pairs']:
            best_pair = st.session_state['cointegrated_pairs'][0]
            asset1, asset2 = best_pair['asset1'], best_pair['asset2']
            
            st.success(f"🏆 **Melhor par encontrado:** {asset1} vs {asset2}")
            
            # Mostrar métricas do par automaticamente selecionado
            col_auto1, col_auto2, col_auto3 = st.columns(3)
            with col_auto1:
                st.metric("Correlação", f"{best_pair['correlation']:.3f}")
            with col_auto2:
                st.metric("P-Value", f"{best_pair['p_value']:.4f}")
            with col_auto3:
                status = "✅ Cointegrado" if best_pair['p_value'] < 0.05 else "❌ Não cointegrado"
                st.metric("Status", status)
        else:
            st.warning("⚠️ Execute a busca de pares primeiro na aba 'Identificar Pares'")
            if len(all_assets) > 1:
                asset1, asset2 = all_assets[0], all_assets[1]
            else:
                asset1, asset2 = all_assets[0], all_assets[0]
    
    # Executar análise se os ativos são diferentes
    if asset1 != asset2:
        _execute_detailed_analysis(pair_analyzer, asset1, asset2, params, show_correlation, show_details)
    else:
        st.error("❌ Selecione dois ativos diferentes para análise")
    
    return asset1, asset2


def _select_assets_for_analysis(analysis_mode, all_assets):
    """Seleciona ativos para análise (manual ou automático)"""
    asset1, asset2 = None, None
    
    if analysis_mode == "✋ Selecionar par manualmente":
        col1, col2 = st.columns(2)
        with col1:
            asset1 = st.selectbox("Primeiro ativo:", all_assets, key="manual_asset1")
        with col2:
            asset2 = st.selectbox("Segundo ativo:", all_assets, key="manual_asset2")
    else:
        # Usar melhor par encontrado
        if 'cointegrated_pairs' in st.session_state and st.session_state['cointegrated_pairs']:
            best_pair = st.session_state['cointegrated_pairs'][0]
            asset1, asset2 = best_pair['asset1'], best_pair['asset2']
            st.info(f"🏆 Analisando melhor par: **{asset1}** vs **{asset2}**")
        else:
            st.warning("⚠️ Execute a busca de pares primeiro na aba 'Identificar Pares'")
            asset1, asset2 = all_assets[0], all_assets[1]
    
    return asset1, asset2


def _execute_detailed_analysis(pair_analyzer, asset1, asset2, params, show_correlation=True, show_details=False):
    """Executa análise detalhada do par"""
    with st.spinner(f"Analisando par {asset1} vs {asset2}..."):
        coint_result = pair_analyzer.test_cointegration(asset1, asset2)
        
        if coint_result:
            # Mostrar métricas básicas sempre
            _display_pair_metrics(coint_result, show_correlation)
            
            # Gráfico de preços normalizados
            _display_normalized_prices_chart(coint_result, asset1, asset2)
            
            # Análise do spread
            _display_spread_analysis(coint_result, asset1, asset2, params)
            
            # Estatísticas detalhadas (opcional)
            if show_details:
                _display_spread_statistics(coint_result, params)
        else:
            st.error("❌ Erro na análise de cointegração do par selecionado")


def _display_pair_metrics(coint_result, show_correlation=True):
    """Exibe métricas principais do par"""
    st.markdown("### 📊 Métricas do Par")
    
    if show_correlation:
        col1, col2, col3, col4 = st.columns(4)
    else:
        col2, col3, col4 = st.columns(3)
    
    if show_correlation:
        with col1:
            correlation = coint_result.get('correlation', 0)
            if 'correlation' not in coint_result:
                # Calcular correlação se não estiver disponível
                price1 = coint_result.get('price1')
                price2 = coint_result.get('price2')
                if price1 is not None and price2 is not None:
                    correlation = price1.corr(price2)
            st.metric("Correlação", f"{correlation:.3f}")
    
    with col2:
        status = "✅ Sim" if coint_result.get('is_cointegrated', False) else "❌ Não"
        st.metric("Cointegrado", status)
    
    with col3:
        st.metric("P-Value", f"{coint_result.get('p_value', 0):.4f}")
    
    with col4:
        st.metric("R²", f"{coint_result.get('r_squared', 0):.3f}")


def _display_normalized_prices_chart(coint_result, asset1, asset2):
    """Exibe gráfico de preços normalizados"""
    price1 = coint_result['price1']
    price2 = coint_result['price2']
    
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


def _display_spread_analysis(coint_result, asset1, asset2, params):
    """Exibe análise do spread"""
    spread = coint_result['spread']
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
    entry_threshold = params['entry_threshold']
    exit_threshold = params['exit_threshold']
    
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


def _display_spread_statistics(coint_result, params):
    """Exibe estatísticas do spread"""
    spread = coint_result['spread']
    spread_mean = coint_result['spread_mean']
    spread_std = coint_result['spread_std']
    z_score = (spread - spread_mean) / spread_std
    
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
        entry_threshold = params['entry_threshold']
        exit_threshold = params['exit_threshold']
        
        if abs(current_z) > entry_threshold:
            signal = "🔴 ENTRADA" if current_z > 0 else "🟢 ENTRADA"
        elif abs(current_z) < exit_threshold:
            signal = "🔵 SAÍDA"
        else:
            signal = "⚪ NEUTRO"
        st.metric("Sinal Atual", signal)


def trading_signals_tab(pair_analyzer, asset1, asset2, params):
    """Aba 3: Sinais de trading"""
    st.subheader("⚡ Sinais de Trading")
    
    if asset1 and asset2 and asset1 != asset2:
        _generate_and_display_signals(pair_analyzer, asset1, asset2, params)
    else:
        st.info(SELECT_VALID_PAIR_MSG)


def _generate_and_display_signals(pair_analyzer, asset1, asset2, params):
    """Gera e exibe sinais de trading"""
    # Gerar sinais de trading
    coint_result = pair_analyzer.test_cointegration(asset1, asset2)
    
    if coint_result:
        signals = pair_analyzer.generate_trading_signals(
            coint_result, 
            entry_threshold=params['entry_threshold'],
            exit_threshold=params['exit_threshold'],
            stop_loss=params['stop_loss']
        )
        
        _display_signal_statistics(signals)
        _display_signals_chart(signals, asset1, asset2, params)
        _display_recent_signals_table(signals)
        


def _display_signal_statistics(signals):
    """Exibe estatísticas dos sinais"""
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


def _display_signals_chart(signals, asset1, asset2, params):
    """Exibe gráfico de sinais"""
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
    entry_threshold = params['entry_threshold']
    exit_threshold = params['exit_threshold']
    
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


def _display_recent_signals_table(signals):
    """Exibe tabela de sinais recentes"""
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


def backtest_tab(pair_analyzer, asset1, asset2, params, df):
    """Aba 4: Backtest da estratégia"""
    st.subheader("📈 Backtest da Estratégia")
    
    if asset1 and asset2 and asset1 != asset2:
        _execute_backtest_analysis(pair_analyzer, asset1, asset2, params, df)
    else:
        st.info("📊 Selecione um par válido na aba 'Análise Detalhada' primeiro.")


def _execute_backtest_analysis(pair_analyzer, asset1, asset2, params, df):
    """Executa análise de backtest"""
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
            "Período de lookback (dias):", 30, 252, 90
        )
    
    if st.button("🚀 Executar Backtest"):
        _run_backtest(pair_analyzer, asset1, asset2, params, df, 
                     initial_capital, lookback_period)


def _run_backtest(pair_analyzer, asset1, asset2, params, df, 
                 initial_capital, lookback_period):
    """Executa o backtest"""
    # pair_analyzer não é utilizado diretamente, mas mantido por compatibilidade de API
    with st.spinner("Executando backtest..."):
        # Executar análise completa com período limitado
        end_date = df.index[-1]
        start_date = end_date - pd.DateOffset(days=lookback_period)
        df_period = df.loc[start_date:end_date]
        
        # Criar analyzer temporário com dados limitados
        from pair_trading import PairTradingAnalysis
        temp_analyzer = PairTradingAnalysis(df_period)
        coint_result = temp_analyzer.test_cointegration(asset1, asset2)
        
        if coint_result:
            signals = temp_analyzer.generate_trading_signals(
                coint_result,
                entry_threshold=params['entry_threshold'],
                exit_threshold=params['exit_threshold'],
                stop_loss=params['stop_loss']
            )
            
            backtest_result = temp_analyzer.backtest_strategy(
                coint_result, signals, 
                transaction_cost=params['transaction_cost']/100
            )
            
            if backtest_result:
                _display_backtest_results(backtest_result, initial_capital, 
                                        coint_result, asset1, asset2)
            else:
                st.error("❌ Erro no backtest")
        else:
            st.error("❌ Erro na análise de cointegração")


def _display_backtest_results(backtest_result, initial_capital, coint_result, asset1, asset2):
    """Exibe resultados do backtest"""
    # Métricas de performance
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(TOTAL_RETURN_LABEL, format_percentage(backtest_result['total_return']))
    with col2:
        st.metric("Retorno Anual", format_percentage(backtest_result['annual_return']))
    with col3:
        st.metric("Sharpe Ratio", format_ratio(backtest_result['sharpe_ratio']))
    with col4:
        st.metric("Max Drawdown", format_percentage(backtest_result['max_drawdown']))
    
    # Segunda linha de métricas
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Volatilidade", format_percentage(backtest_result['annual_volatility']))
    with col2:
        st.metric("Número de Trades", f"{backtest_result['num_trades']}")
    with col3:
        win_rate = backtest_result.get('win_rate', 0)
        st.metric("Win Rate", format_percentage(win_rate))
    with col4:
        capital_final = initial_capital * (1 + backtest_result['total_return'])
        st.metric(FINAL_CAPITAL_LABEL, f"R$ {capital_final:,.0f}")
    
    # Gráfico de performance
    _display_performance_chart(backtest_result, initial_capital, coint_result, asset1, asset2)
    
    # Interpretação dos resultados
    _display_results_interpretation(backtest_result)


def _display_performance_chart(backtest_result, initial_capital, coint_result, asset1, asset2):
    """Exibe gráfico de performance"""
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


def _display_results_interpretation(backtest_result):
    """Exibe interpretação dos resultados"""
    st.subheader(RESULTS_INTERPRETATION_LABEL)
    
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


def optimization_tab(pair_analyzer, asset1, asset2, params):
    """Aba 5: Otimização de parâmetros"""
    st.subheader("🎯 Otimização de Parâmetros")
    st.info("💡 Encontre os melhores parâmetros para maximizar o Sharpe Ratio")
    
    if asset1 and asset2 and asset1 != asset2:
        _execute_parameter_optimization(pair_analyzer, asset1, asset2, params)
    else:
        st.info("📊 Selecione um par válido na aba 'Análise Detalhada' primeiro.")


def _execute_parameter_optimization(pair_analyzer, asset1, asset2, params):
    """Executa otimização de parâmetros"""
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
        _run_optimization(pair_analyzer, asset1, asset2, params, 
                         entry_range, exit_range, optimization_step)


def _evaluate_parameter_combination(pair_analyzer, coint_result, entry_thresh, exit_thresh, params):
    """Avalia uma combinação específica de parâmetros"""
    # Gerar sinais
    signals = pair_analyzer.generate_trading_signals(
        coint_result,
        entry_threshold=entry_thresh,
        exit_threshold=exit_thresh,
        stop_loss=params['stop_loss']
    )
    
    # Backtest
    backtest = pair_analyzer.backtest_strategy(
        coint_result, signals,
        transaction_cost=params['transaction_cost']/100
    )
    
    if not backtest:
        return None
        
    return {
        'entry_threshold': entry_thresh,
        'exit_threshold': exit_thresh,
        'sharpe_ratio': backtest['sharpe_ratio'],
        'total_return': backtest['total_return'],
        'max_drawdown': backtest['max_drawdown'],
        'num_trades': backtest['num_trades']
    }


def _perform_parameter_optimization(pair_analyzer, coint_result, entry_values, exit_values, params, progress_bar, total_combinations):
    """Executa o loop de otimização de parâmetros"""
    best_sharpe = -np.inf
    best_params = {}
    optimization_results = []
    current_combination = 0
    
    # Loop de otimização
    for entry_thresh in entry_values:
        for exit_thresh in exit_values:
            # Apenas avaliar combinações válidas (exit < entry)
            if exit_thresh < entry_thresh:
                # Avaliar esta combinação de parâmetros
                result = _evaluate_parameter_combination(
                    pair_analyzer, coint_result, entry_thresh, exit_thresh, params
                )
                
                # Processar resultado se válido
                if result:
                    optimization_results.append(result)
                    # Verificar se é o melhor resultado até agora
                    if result['sharpe_ratio'] > best_sharpe:
                        best_sharpe = result['sharpe_ratio']
                        best_params = {
                            'entry_threshold': entry_thresh,
                            'exit_threshold': exit_thresh,
                            'backtest': {
                                'sharpe_ratio': result['sharpe_ratio'],
                                'total_return': result['total_return'],
                                'max_drawdown': result['max_drawdown'],
                                'num_trades': result['num_trades']
                            }
                        }
            
            # Atualizar progresso
            current_combination += 1
            progress_bar.progress(current_combination / total_combinations)
    
    return best_params, best_sharpe, optimization_results


def _run_optimization(pair_analyzer, asset1, asset2, params, 
                     entry_range, exit_range, optimization_step):
    """Executa a otimização dos parâmetros"""
    with st.spinner("Otimizando parâmetros..."):
        # Preparar valores para otimização
        entry_values = np.arange(entry_range[0], entry_range[1] + optimization_step, optimization_step)
        exit_values = np.arange(exit_range[0], exit_range[1] + optimization_step, optimization_step)
        
        # Inicializar barra de progresso
        total_combinations = len(entry_values) * len(exit_values)
        progress_bar = st.progress(0)
        
        # Testar cointegração uma única vez
        coint_result = pair_analyzer.test_cointegration(asset1, asset2)
        
        # Executar otimização
        best_params, best_sharpe, optimization_results = _perform_parameter_optimization(
            pair_analyzer, coint_result, entry_values, exit_values, params, progress_bar, total_combinations
        )
        
        # Exibir resultados
        if best_params:
            _display_optimization_results(best_params, best_sharpe, optimization_results)
        else:
            st.warning("⚠️ Nenhuma combinação válida encontrada")


def _display_optimization_results(best_params, best_sharpe, optimization_results):
    """Exibe resultados da otimização"""
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
        _display_optimization_heatmap(optimization_results)


def _display_optimization_heatmap(optimization_results):
    """Exibe heatmap dos resultados de otimização"""
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


def tutorial_tab(all_assets=None):
    """Aba 6: Tutorial completo"""
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
    6. **Implementação**: Execute com capital real (comece pequeno!)
    
    **Lembre-se**: Pair Trading requer disciplina, paciência e gestão de risco rigorosa!
    """)
      # Sugestões inteligentes de pares baseada em categorias de ativos
    st.subheader("💡 Sugestões Inteligentes de Pares")
    st.info("Baseado em categorias de ativos, aqui estão algumas sugestões de pares para considerar:")

    # Passar all_assets apenas se não for None
    smart_suggestions = _get_smart_pair_suggestions(all_assets)
    
    if smart_suggestions:
        for suggestion in smart_suggestions:
            st.markdown(f"- **{suggestion['asset1']}** e **{suggestion['asset2']}**: {suggestion['reason']}")
    else:
        st.info("Nenhuma sugestão disponível com os ativos selecionados.")


def _get_smart_pair_suggestions(all_assets=None):
    """Retorna sugestões inteligentes de pares baseadas em categorias"""
    suggestions = []
    
    if all_assets is None:
        all_assets = []
    
    # Sugestões por categoria
    category_suggestions = {
        "Bancos": ["ITUB4.SA", "BBDC4.SA", "BBAS3.SA", "SANB11.SA"],
        "Petróleo & Mineração": ["PETR4.SA", "VALE3.SA", "USIM5.SA", "GOAU4.SA"],
        "Consumo": ["ABEV3.SA", "JBSS3.SA", "LREN3.SA", "MGLU3.SA"],
        "Utilities": ["ELET3.SA", "CSAN3.SA", "VIVT3.SA"],
        "ETFs": ["BOVA11.SA", "SMAL11.SA", "IVVB11.SA"],
        "Criptomoedas": ["BTC-USD", "ETH-USD", "ADA-USD", "SOL-USD", "XRP-USD"],
        "Forex": ["USDBRL=X", "EURBRL=X", "JPY=X", "USDJPY=X"]
    }
    
    for category, assets in category_suggestions.items():
        available_in_category = [a for a in assets if a in all_assets]
        if len(available_in_category) >= 2:
            # Adicionar combinações dessa categoria
            for i in range(len(available_in_category)):
                for j in range(i+1, len(available_in_category)):
                    suggestions.append({
                        'asset1': available_in_category[i],
                        'asset2': available_in_category[j],
                        'category': category,
                        'reason': f"Mesmo setor: {category}"
                    })
    
    # Sugestões cross-category (diferentes setores para diversificação)
    cross_suggestions = [
        {"assets": ["PETR4.SA", "ITUB4.SA"], "reason": "Energia vs Bancário"},
        {"assets": ["VALE3.SA", "WEGE3.SA"], "reason": "Commodities vs Industrial"},
        {"assets": ["BTC-USD", "USDBRL=X"], "reason": "Cripto vs Forex"},
        {"assets": ["BOVA11.SA", "^BVSP"], "reason": "ETF vs Índice"},
    ]
    
    for suggestion in cross_suggestions:
        assets = suggestion["assets"]
        if all(a in all_assets for a in assets):
            suggestions.append({
                'asset1': assets[0],
                'asset2': assets[1],
                'category': 'Cross-Sector',
                'reason': suggestion["reason"]
            })
    
    return suggestions[:10]  # Retornar top 10 sugestões
