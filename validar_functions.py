import streamlit as st
import pandas as pd
import numpy as np
import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def validar_funcoes():
    """Script para validar as funções corrigidas"""
    st.title("Validação das Funções Corrigidas")
    
    # Gerar dados sintéticos
    np.random.seed(42)
    n_samples = 100
    n_features = 10
    
    # Dados sintéticos
    data = np.random.randn(n_samples, n_features)
    start_date = datetime.datetime(2023, 1, 1)
    dates = [start_date + datetime.timedelta(days=i) for i in range(n_samples)]
    
    # Criar DataFrame de teste
    columns = [f"Asset_{i+1}" for i in range(n_features)]
    df = pd.DataFrame(data, index=dates, columns=columns)
    
    # Tab para cada tipo de validação
    tabs = st.tabs([
        "1. Pair Trading",
        "2. Advanced PCA",
        "3. Statistical Analysis"
    ])
    
    # Tab 1: Validar Pair Trading Functions
    with tabs[0]:
        st.header("Pair Trading Functions")
        try:
            from pair_trading_helpers import (
                _display_correlation_matrix, find_cointegrated_pairs_tab, 
                _select_assets_for_analysis
            )
            from pair_trading import PairTradingAnalysis
            
            st.write("### Testando _display_correlation_matrix")
            # Teste com selected_assets vazio
            st.write("Teste 1: selected_assets vazio")
            with st.expander("Ver resultado"):
                _display_correlation_matrix([], df)
                
            # Teste com selected_assets definido
            st.write("Teste 2: selected_assets definido")
            with st.expander("Ver resultado"):
                _display_correlation_matrix(df.columns.tolist()[:3], df)
                
            st.success("✓ _display_correlation_matrix funciona corretamente")
            
            # Não testamos outras funções que precisam de interação do usuário
            st.info("Para outras funções de Pair Trading, teste diretamente no app principal")
            
        except Exception as e:
            st.error(f"Erro ao validar Pair Trading functions: {str(e)}")
    
    # Tab 2: Validar Advanced PCA Functions
    with tabs[1]:
        st.header("Advanced PCA Functions")
        try:
            from advanced_pca_simplificado import (
                execute_static_pca_analysis, execute_rolling_pca_analysis,
                display_portfolio_results
            )
            
            st.write("### Testando execute_static_pca_analysis")
            with st.expander("Ver resultado"):
                # Criar returns
                returns = df.pct_change().dropna()
                selected_assets = df.columns.tolist()[:5]
                pca, components, explained_var = execute_static_pca_analysis(
                    returns, selected_assets, 3
                )
                st.write("PCA executado com sucesso!")
                
            st.write("### Testando execute_rolling_pca_analysis")
            with st.expander("Ver resultado"):
                rolling_variance, rolling_loadings_pc1, rolling_dates = execute_rolling_pca_analysis(
                    returns, selected_assets, 30, "Mensal", 10
                )
                st.write("PCA rolling executado com sucesso!")
                
            st.success("✓ Funções de Advanced PCA funcionam corretamente")
            
        except Exception as e:
            st.error(f"Erro ao validar Advanced PCA functions: {str(e)}")
    
    # Tab 3: Validar Statistical Analysis Functions
    with tabs[2]:
        st.header("Statistical Analysis Functions")
        try:
            from statistical_analysis_robusto import StatisticalAnalysis
            
            st.write("### Testando StatisticalAnalysis")
            with st.expander("Ver resultado"):
                # Criar objeto
                stat_analysis = StatisticalAnalysis(df)
                st.write("StatisticalAnalysis criado com sucesso!")
                
                # Testar comparação
                comparison = {
                    'data1': df['Asset_1'].values,
                    'data2': df['Asset_2'].values,
                    'name1': 'Asset 1',
                    'name2': 'Asset 2',
                    'comparison_tests': {}
                }
                
                st.write("Testando display_comparison_results:")
                stat_analysis.display_comparison_results(comparison)
                
            st.success("✓ Funções de Statistical Analysis funcionam corretamente")
            
        except Exception as e:
            st.error(f"Erro ao validar Statistical Analysis functions: {str(e)}")

if __name__ == "__main__":
    validar_funcoes()
