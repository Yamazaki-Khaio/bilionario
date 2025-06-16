"""
Script temporário para usar a versão simplificada do PCA
Este script é uma versão modificada do app.py original, mas usando advanced_pca_simplificado.py
"""

import os
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

# Importações do projeto
from constants import *
from app_helpers import get_raw_data_path, load_css

# Configuração da página
st.set_page_config(
    page_title="Bilionário App - Versão Simplificada",
    layout="wide",
    page_icon="💰",
)

# Carregar CSS personalizado
load_css()

def show_advanced_pca_page():
    """Página de PCA avançado com a versão simplificada"""
    # MODIFICAÇÃO AQUI: Usar o módulo simplificado em vez do original
    from advanced_pca_simplificado import (
        setup_pca_sidebar, execute_static_pca_analysis, display_pca_loadings,
        execute_rolling_pca_analysis, display_rolling_pca_stability, build_pca_portfolio,
        display_portfolio_results, analyze_pca_risk
    )
    
    st.title("📊 Análise PCA Avançada (Versão Simplificada)")
    
    RAW_DATA = get_raw_data_path()
    
    if not os.path.exists(RAW_DATA):
        st.error("Dados não encontrados. Por favor, gere os dados primeiro.")
        return
    
    try:
        df = pd.read_csv(RAW_DATA, index_col=0, parse_dates=True)
        returns = df.pct_change().dropna()
        
        # Configuração do sidebar
        sidebar_config = setup_pca_sidebar(returns)
        
        if sidebar_config is None:
            st.warning("Selecione pelo menos 3 ativos para análise PCA.")
            return
        
        selected_assets, n_components, rebalance_freq, window_size, rebalance_window = sidebar_config
        returns_selected = returns[selected_assets]
        
        st.subheader("🚀 PCA Estático")
        
        # Análise PCA estática
        pca, components, explained_var = execute_static_pca_analysis(returns_selected, selected_assets, n_components)
        
        # Exibir loadings
        display_pca_loadings(pca, selected_assets, explained_var)
        
        # Análise PCA rolling
        rolling_results = execute_rolling_pca_analysis(
            returns_selected, selected_assets, window_size, rebalance_freq, rebalance_window
        )
        rolling_variance, rolling_loadings_pc1, rolling_dates = rolling_results
        
        # Exibir estabilidade rolling
        if len(rolling_dates) > 0:
            display_rolling_pca_stability(
                rolling_variance, rolling_loadings_pc1, rolling_dates, selected_assets
            )
            
            st.subheader("🎯 Estratégias de Portfolio Baseadas em PCA")
            
            # Selector de estratégia
            strategy_type = st.selectbox(
                "Selecione estratégia de portfolio:",
                ["Maximum Diversification", "Minimum Variance", "Equal Risk Contribution"]
            )
            
            # Construir portfolio baseado em PCA
            portfolio_results = build_pca_portfolio(
                pca, components, returns_selected, selected_assets, strategy_type, rebalance_freq
            )
            weights, portfolio_cumulative, total_return, annual_return, annual_vol, sharpe = portfolio_results
            
            # Exibir resultados do portfolio
            display_portfolio_results(
                weights, portfolio_cumulative, total_return, annual_return, annual_vol, sharpe, selected_assets, strategy_type
            )
            
            # Análise de risco PCA
            analyze_pca_risk(pca, selected_assets, n_components)
            
            st.subheader("📚 Recursos Educacionais")
            display_interactive_pca_example(pca, selected_assets, n_components)
            display_pca_educational_content()
        
    except Exception as e:
        st.error(f"Erro ao processar dados PCA: {str(e)}")

if __name__ == "__main__":
    show_advanced_pca_page()
