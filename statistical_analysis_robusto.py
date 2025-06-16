"""
Versão corrigida do statistical_analysis.py com melhor tratamento de erros
Foca em robustez e tratamento de casos extremos
"""

import pandas as pd
import numpy as np
import streamlit as st
from scipy import stats
from scipy.stats import skew, kurtosis, jarque_bera, shapiro
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from scipy.stats import linregress, pearsonr, spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from financial_formatting import format_percentage, format_ratio
from statistical_analysis_helpers import (
    plot_scatter_chart,
    plot_histogram_comparison, 
    plot_box_comparison,
    plot_qq_comparison,
    display_distribution_comparison_metrics,
    plot_correlation_heatmap,
)

class StatisticalAnalyzer:
    """Classe para análise estatística de ativos financeiros"""
    
    def find_different_distributions(self, returns, min_obs=100, top_n=5):
        """
        Encontra pares de ativos com distribuições mais diferentes
        
        Args:
            returns (pd.DataFrame): DataFrame com retornos dos ativos
            min_obs (int): Número mínimo de observações para análise
            top_n (int): Número de pares a retornar
            
        Returns:
            list: Lista com os top_n pares de distribuições mais diferentes
        """
        # Busca de ativos com distribuições estatisticamente diferentes
        st.subheader("🔍 Distribuições Estatisticamente Diferentes")
        
        # Verificar se há dados suficientes
        if returns.shape[0] < min_obs:
            st.warning(f"Dados insuficientes. A análise requer pelo menos {min_obs} observações, mas encontrou {returns.shape[0]}.")
            return []
        
        pairs = []
        max_difference = 0
        progress_bar = st.progress(0)
        assets = returns.columns
        total_combinations = len(assets) * (len(assets) - 1) // 2
        counter = 0
        
        try:
            with st.spinner("Buscando pares com distribuições estatisticamente diferentes..."):
                for i, asset1 in enumerate(assets):
                    for j, asset2 in enumerate(assets[i+1:], i+1):
                        # Atualizar barra de progresso
                        counter += 1
                        progress_bar.progress(counter / total_combinations)
                        
                        # Pular se não tiver dados suficientes (pelo menos min_obs/2 em cada)
                        aligned_data = pd.concat([returns[asset1], returns[asset2]], axis=1).dropna()
                        if aligned_data.shape[0] < min_obs/2:
                            continue
                        
                        aligned_data1, aligned_data2 = aligned_data[asset1], aligned_data[asset2]
                        
                        # Testes estatísticos
                        try:
                            comparison = self._compare_distributions(
                                aligned_data1, aligned_data2, asset1, asset2
                            )
                            
                            # Calcular diferença total (combinação de vários testes)
                            try:
                                # Verificar se comparison_tests e ks_test existem antes de acessá-los
                                if ('comparison_tests' in comparison and 
                                    'ks_test' in comparison['comparison_tests'] and 
                                    'mann_whitney' in comparison['comparison_tests']):
                                    
                                    difference_score = (
                                        (1 - comparison['comparison_tests']['ks_test'].get('p_value', 0.5)) * 0.3 +
                                        (1 - comparison['comparison_tests']['mann_whitney'].get('p_value', 0.5)) * 0.2 +
                                        abs(comparison['statistics']['asset1'].get('skewness', 0) - 
                                            comparison['statistics']['asset2'].get('skewness', 0)) * 0.2 +
                                        abs(comparison['statistics']['asset1'].get('kurtosis', 0) - 
                                            comparison['statistics']['asset2'].get('kurtosis', 0)) * 0.3
                                    )
                                else:
                                    difference_score = 0.0
                            except Exception as e:
                                st.warning(f"Erro ao calcular diferença entre {asset1} e {asset2}: {str(e)}")
                                difference_score = 0.0
                            
                            # Adicionar à lista de pares
                            pairs.append({
                                'asset1': asset1,
                                'asset2': asset2,
                                'difference_score': difference_score,
                                'comparison': comparison
                            })
                            
                        except Exception as e:
                            st.warning(f"Erro ao comparar {asset1} e {asset2}: {str(e)}")
                
            # Ordenar por diferença e retornar top_n
            pairs.sort(key=lambda x: x['difference_score'], reverse=True)
            return pairs[:top_n]
            
        except Exception as e:
            st.error(f"Erro na análise de distribuições: {str(e)}")
            return []
            
    def _compare_distributions(self, data1, data2, name1, name2):
        """
        Compara distribuições entre dois ativos
        
        Args:
            data1, data2: Séries de retornos
            name1, name2: Nomes dos ativos
            
        Returns:
            dict: Resultados detalhados da comparação
        """
        try:
            # Estatísticas descritivas
            stats1 = {
                'mean': data1.mean(),
                'std': data1.std(),
                'skewness': skew(data1),
                'kurtosis': kurtosis(data1, fisher=True),
                'min': data1.min(),
                'max': data1.max(),
                'count': len(data1)
            }
            
            stats2 = {
                'mean': data2.mean(),
                'std': data2.std(),
                'skewness': skew(data2),
                'kurtosis': kurtosis(data2, fisher=True),
                'min': data2.min(),
                'max': data2.max(),
                'count': len(data2)
            }
            
            # Testes de normalidade
            try:
                shapiro1 = shapiro(data1.values)
                shapiro2 = shapiro(data2.values)
                jb1 = jarque_bera(data1.values)
                jb2 = jarque_bera(data2.values)
            except Exception as e:
                # Em caso de erro, usar valores padrão
                shapiro1 = (0, 1.0)
                shapiro2 = (0, 1.0)
                jb1 = (0, 1.0)
                jb2 = (0, 1.0)
                
            # Verificar tamanho mínimo para executar testes estatísticos
            min_observations = 30  # Normalmente se considera 30 como mínimo para testes paramétricos

            # Inicializar valores padrão
            ks_stat, ks_p = 0, 1.0
            mw_stat, mw_p = 0, 1.0
            levene_stat, levene_p = 0, 1.0
            t_stat, t_p = 0, 1.0
            
            # Só executar se tiver dados suficientes
            if len(data1.values) >= min_observations and len(data2.values) >= min_observations:
                try:
                    # Teste de Kolmogorov-Smirnov (distribuições diferentes)
                    ks_stat, ks_p = stats.ks_2samp(data1.values, data2.values)
                except Exception:
                    pass
                
                try:
                    # Teste Mann-Whitney U (medianas diferentes)
                    mw_stat, mw_p = stats.mannwhitneyu(data1.values, data2.values, alternative='two-sided')
                except Exception:
                    pass
                    
                try:
                    # Teste Levene (variâncias diferentes)
                    levene_stat, levene_p = stats.levene(data1.values, data2.values)
                except Exception:
                    pass
                    
                try:
                    # Teste t para médias (assumindo normalidade)
                    t_stat, t_p = stats.ttest_ind(data1.values, data2.values)
                except Exception:
                    pass
            
            # Construir resultado com tratamento adequado de erros
            return {
                'assets': {'asset1': name1, 'asset2': name2},
                'statistics': {'asset1': stats1, 'asset2': stats2},
                'normality_tests': {
                    'asset1': {'shapiro': shapiro1, 'jarque_bera': jb1},
                    'asset2': {'shapiro': shapiro2, 'jarque_bera': jb2}
                },
                'comparison_tests': {
                    'ks_test': {'statistic': ks_stat, 'p_value': ks_p, 
                            'significant': ks_p < 0.05},
                    'mann_whitney': {'statistic': mw_stat, 'p_value': mw_p,
                                'significant': mw_p < 0.05},
                    'levene_test': {'statistic': levene_stat, 'p_value': levene_p,
                                'significant': levene_p < 0.05},
                    't_test': {'statistic': t_stat, 'p_value': t_p,
                            'significant': t_p < 0.05}
                }
            }
        except Exception as e:
            # Em caso de qualquer erro, retornar uma estrutura com valores padrão
            # para evitar quebrar o app
            st.warning(f"Erro ao comparar distribuições: {str(e)}")
            return {
                'assets': {'asset1': name1, 'asset2': name2},
                'statistics': {'asset1': {'mean': 0, 'std': 0}, 'asset2': {'mean': 0, 'std': 0}},
                'comparison_tests': {
                    'ks_test': {'statistic': 0, 'p_value': 1.0, 'significant': False},
                    'mann_whitney': {'statistic': 0, 'p_value': 1.0, 'significant': False},
                    'levene_test': {'statistic': 0, 'p_value': 1.0, 'significant': False},
                    't_test': {'statistic': 0, 'p_value': 1.0, 'significant': False}
                }
            }
