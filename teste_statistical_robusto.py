"""
Script para testar a versão robusta do statistical_analysis
"""
import streamlit as st
import pandas as pd
import numpy as np
import os

# Configuração
st.set_page_config(
    page_title="Teste Statistical Analysis Robusto",
    layout="wide",
)

st.title("🔬 Teste de Análise Estatística Robusta")

# Carregar dados
@st.cache_data
def get_raw_data():
    """Carrega dados brutos ou gera dados aleatórios"""
    # Se arquivo existe, carregar
    if os.path.exists("data/raw_data.csv"):
        df = pd.read_csv("data/raw_data.csv", index_col=0, parse_dates=True)
    else:
        # Criar dados de teste
        st.info("Arquivo de dados não encontrado. Criando dados aleatórios para teste.")
        np.random.seed(42)
        
        index = pd.date_range(start="2020-01-01", periods=252, freq="B")
        assets = ["ATIVO"+str(i) for i in range(1, 11)]
        
        # Criar retornos com diferentes distribuições
        df = pd.DataFrame(index=index)
        for i, asset in enumerate(assets):
            # Criar diferentes distribuições para testar
            if i % 3 == 0:  # Normal
                df[asset] = np.random.normal(0.001, 0.02, len(index))
            elif i % 3 == 1:  # Assimétrica
                df[asset] = np.random.lognormal(0, 0.02, len(index)) - 1
            else:  # Cauda pesada
                df[asset] = np.random.standard_t(3, size=len(index)) * 0.01
                
    return df

# Carregar dados e calcular retornos
df = get_raw_data()
returns = df.pct_change().dropna()

# Exibir dados
st.subheader("📊 Dados de Teste")
st.dataframe(returns.head())

# Importar e testar a versão robusta
try:
    from statistical_analysis_robusto import StatisticalAnalyzer
    
    st.subheader("🧪 Testando Análise Estatística Robusta")
    
    analyzer = StatisticalAnalyzer()
    
    # Testar com diferentes números de observações
    test_sizes = [30, 50, 100, 200]
    
    for size in test_sizes:
        st.write(f"### Teste com {size} observações")
        sample_returns = returns.iloc[-size:] if len(returns) >= size else returns
        
        # Testar encontrar distribuições diferentes
        with st.expander(f"Resultado com {size} observações"):
            different_pairs = analyzer.find_different_distributions(sample_returns, min_obs=size//2, top_n=3)
            
            if different_pairs:
                st.success(f"✅ Sucesso! Encontrados {len(different_pairs)} pares com distribuições diferentes")
                
                # Mostrar o primeiro par como exemplo
                if different_pairs:
                    pair = different_pairs[0]
                    st.write(f"**Par mais diferente**: {pair['asset1']} vs {pair['asset2']}")
                    st.write(f"**Score de diferença**: {pair['difference_score']:.4f}")
            else:
                st.info("Nenhum par encontrado ou dados insuficientes")
                
except Exception as e:
    st.error(f"Erro ao testar a análise estatística: {str(e)}")
    st.code(f"Detalhes do erro: {str(e)}", language="python")
