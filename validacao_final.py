import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import os
import sys
import datetime

# Configuração básica do Streamlit
st.set_page_config(
    page_title="Validação Final",
    page_icon="✓",
    layout="wide"
)

# Importa as funções que queremos testar
try:
    from advanced_pca_simplificado import (
        execute_static_pca_analysis,
        display_pca_loadings,
        execute_rolling_pca_analysis,
        display_interactive_pca_example,
        build_pca_portfolio,
        analyze_pca_risk
    )
    st.success("✓ Módulo advanced_pca_simplificado importado com sucesso")
except Exception as e:
    st.error(f"❌ Erro ao importar advanced_pca_simplificado: {str(e)}")

try:
    from statistical_analysis_robusto import (
        StatisticalAnalysis
    )
    st.success("✓ Módulo statistical_analysis_robusto importado com sucesso")
except Exception as e:
    st.error(f"❌ Erro ao importar statistical_analysis_robusto: {str(e)}")

st.title("✓ Validação Final das Correções")
st.write("Este script valida todas as correções feitas no aplicativo Bilionário.")

# Cria dados de teste simples
np.random.seed(42)
n_samples = 100
n_features = 5

# Dados sintéticos
data = np.random.randn(n_samples, n_features)
selected_assets = ["Asset_" + str(i) for i in range(1, n_features+1)]

# Datas para o índice
start_date = datetime.datetime(2023, 1, 1)
dates = [start_date + datetime.timedelta(days=i) for i in range(n_samples)]

# Cria DataFrame com retornos
returns_df = pd.DataFrame(
    data, 
    columns=selected_assets,
    index=dates
)

# Testes de funções corrigidas
st.header("1. Teste de PCA Estático")
try:
    pca, loadings, n_components = execute_static_pca_analysis(returns_df, selected_assets)
    st.success("✓ execute_static_pca_analysis funcionou corretamente")
except Exception as e:
    st.error(f"❌ execute_static_pca_analysis falhou: {str(e)}")
    st.code(str(e))

st.header("2. Teste do Display Interativo")
try:
    with st.expander("Visualizar teste do display interativo"):
        if 'pca' in locals() and pca is not None:
            display_interactive_pca_example(pca, selected_assets, n_components)
            st.success("✓ display_interactive_pca_example funcionou corretamente")
        else:
            st.warning("⚠️ Não foi possível executar display_interactive_pca_example porque o PCA falhou")
except Exception as e:
    st.error(f"❌ display_interactive_pca_example falhou: {str(e)}")
    st.code(str(e))

st.header("3. Teste do PCA Rolling")
try:
    with st.expander("Visualizar teste do PCA Rolling"):
        result_rolling = execute_rolling_pca_analysis(returns_df, selected_assets)
        st.success("✓ execute_rolling_pca_analysis funcionou corretamente")
except Exception as e:
    st.error(f"❌ execute_rolling_pca_analysis falhou: {str(e)}")
    st.code(str(e))

st.header("4. Teste da Análise Estatística Robusta")
try:
    with st.expander("Visualizar teste da análise estatística"):
        # Criando dados de teste para a análise estatística
        comparison = {
            'data1': data[:50, 0],
            'data2': data[50:, 0],
            'name1': 'Group A',
            'name2': 'Group B',
            'comparison_tests': {}  # Inicializa como vazio para testar a robustez
        }
        
        # Instancia a classe StatisticalAnalysis
        stat_analysis = StatisticalAnalysis()
        
        # Tenta exibir a comparação, que deveria funcionar mesmo sem 'ks_test' definido
        st.write("Teste de robustez com dictionary incompleto:")
        stat_analysis.display_comparison_results(comparison)
        
        st.success("✓ Análise estatística robusta funcionou corretamente")
except Exception as e:
    st.error(f"❌ Análise estatística robusta falhou: {str(e)}")
    st.code(str(e))

st.header("Resultado Geral")
st.write("Todos os testes foram concluídos. Verifique os resultados acima para confirmar que todas as correções foram aplicadas corretamente.")
