#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STATISTICAL ANALYSIS HELPERS - FUNÇÕES AUXILIARES
==================================================
Módulo contendo funções auxiliares para reduzir a complexidade cognitiva
da função show_statistical_analysis_page().
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from scipy.stats import skew, kurtosis
from plotly.subplots import make_subplots
from constants import PETR4_SYMBOL
import statsmodels.api as sm
# Importar funções de interpretação de risco diretamente do módulo risk_utils.py
from risk_utils import _display_risk_interpretation, _calculate_risk_score, _get_risk_category


def validate_data_for_operations(data, operation_name="estatística", min_samples=30, check_columns=None):
    """
    Função de validação abrangente para verificar dados antes de operações estatísticas.
    
    Parâmetros:
    -----------
    data : DataFrame
        DataFrame a ser validado
    operation_name : str, default="estatística"
        Nome da operação para mensagens de erro
    min_samples : int, default=30
        Número mínimo de amostras necessárias
    check_columns : list, default=None
        Lista de colunas específicas a serem verificadas
        
    Retorno:
    --------
    tuple
        (is_valid, message) - boolean indicando se os dados são válidos e mensagem descritiva
    """
    # Verificações básicas
    if data is None:
        return False, f"❌ Dados não fornecidos para operação {operation_name}"
    
    if not isinstance(data, pd.DataFrame):
        return False, f"❌ Tipo de dados inválido para {operation_name} (esperado DataFrame)"
    
    if data.empty:
        return False, f"❌ DataFrame vazio para operação {operation_name}"
    
    # Verificar tamanho dos dados
    if len(data) < min_samples:
        return False, f"⚠️ Dados insuficientes para {operation_name} ({len(data)} < {min_samples} mínimo recomendado)"
    
    # Se colunas específicas foram especificadas
    if check_columns:
        # Verificar existência das colunas
        missing_columns = [col for col in check_columns if col not in data.columns]
        if missing_columns:
            return False, f"❌ Colunas necessárias não encontradas: {', '.join(missing_columns)}"
        
        # Verificar valores ausentes nas colunas específicas
        na_counts = data[check_columns].isna().sum()
        if na_counts.sum() > 0:
            na_info = ", ".join([f"{col}: {count}" for col, count in na_counts.items() if count > 0])
            if na_counts.sum() > len(data) * 0.1:  # Mais de 10% dos dados são NaN
                return False, f"❌ Muitos valores ausentes: {na_info}"
            else:
                return True, f"⚠️ Alguns valores ausentes: {na_info} (os cálculos podem ser impactados)"
        
        # Verificar valores constantes (variância zero)
        zero_variance = []
        for col in check_columns:
            if data[col].std() < 1e-8:  # Praticamente constante
                zero_variance.append(col)
        
        if zero_variance:
            return False, f"⚠️ Colunas com variação quase zero: {', '.join(zero_variance)}"
    
    # Verificar valores ausentes em geral
    total_na = data.isna().sum().sum()
    if total_na > 0:
        na_percent = total_na / (len(data) * len(data.columns)) * 100
        if na_percent > 10:  # Mais de 10% dos dados são NaN
            return False, f"⚠️ Alto percentual de valores ausentes: {na_percent:.1f}% do total"
        else:
            return True, f"ℹ️ {total_na} valores ausentes ({na_percent:.1f}% do total) - os cálculos podem ser impactados"
    
    return True, "✅ Dados válidos para operação"


def plot_scatter_chart(data1, data2, name1, name2, add_regression=True):
    """
    Plota um gráfico de dispersão (scatter) entre duas séries de dados.
    
    Parâmetros:
    -----------
    data1 : array ou Series
        Primeira série de dados
    data2 : array ou Series
        Segunda série de dados
    name1 : str
        Nome da primeira série (eixo x)
    name2 : str
        Nome da segunda série (eixo y)
    add_regression : bool, default=True
        Se True, adiciona uma linha de regressão
        
    Retorno:
    --------
    None
    """
    try:
        # Verificar se há dados suficientes
        if isinstance(data1, (pd.Series, np.ndarray)) and len(data1) == 0:
            st.warning(f"Não há dados disponíveis para {name1}")
            return
            
        if isinstance(data2, (pd.Series, np.ndarray)) and len(data2) == 0:
            st.warning(f"Não há dados disponíveis para {name2}")
            return
            
        # Criar DataFrame para plotly
        scatter_df = pd.DataFrame({
            name1: data1,
            name2: data2
        }).dropna()
        
        # Verificar se há dados após remoção de valores NaN
        if scatter_df.empty or len(scatter_df) < 3:
            st.warning(f"Dados insuficientes para criar gráfico de dispersão entre {name1} e {name2}")
            return
            
        # Criar gráfico de dispersão
        fig = px.scatter(
            scatter_df, x=name1, y=name2, 
            title=f"Gráfico de Dispersão: {name1} vs {name2}",
            trendline="ols" if add_regression and len(scatter_df) >= 3 else None
        )
        
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Erro ao criar gráfico de dispersão: {str(e)}")
        st.info("Verifique se os dados são válidos e têm formato compatível.")


def plot_histogram_comparison(data1, data2, name1, name2, bins=30):
    """
    Plota histogramas comparativos de duas séries de dados.
    
    Parâmetros:
    -----------
    data1 : array ou Series
        Primeira série de dados
    data2 : array ou Series
        Segunda série de dados
    name1 : str
        Nome da primeira série
    name2 : str
        Nome da segunda série
    bins : int, default=30
        Número de bins para os histogramas
        
    Retorno:
    --------
    None
    """
    try:
        # Verificar se há dados suficientes
        data1_clean = pd.Series(data1).dropna()
        data2_clean = pd.Series(data2).dropna()
        
        if len(data1_clean) == 0 and len(data2_clean) == 0:
            st.warning("Não há dados disponíveis para criar histogramas")
            return
        
        # Garantir um número mínimo de bins se houver poucos dados
        effective_bins1 = min(bins, max(5, len(data1_clean) // 5)) if len(data1_clean) > 0 else bins
        effective_bins2 = min(bins, max(5, len(data2_clean) // 5)) if len(data2_clean) > 0 else bins
        
        fig = go.Figure()
        
        # Adicionar histograma para data1 se houver dados
        if len(data1_clean) > 0:
            fig.add_trace(go.Histogram(
                x=data1_clean,
                name=name1,
                opacity=0.7,
                nbinsx=effective_bins1,
                marker_color='#1f77b4'
            ))
        else:
            st.info(f"Não há dados disponíveis para {name1}")
        
        # Adicionar histograma para data2 se houver dados
        if len(data2_clean) > 0:
            fig.add_trace(go.Histogram(
                x=data2_clean,
                name=name2,
                opacity=0.7,
                nbinsx=effective_bins2,
                marker_color='#ff7f0e'
            ))
        else:
            st.info(f"Não há dados disponíveis para {name2}")
    
        # Se não temos dados para nenhum dos dois, retornamos
        if len(data1_clean) == 0 and len(data2_clean) == 0:
            return
      # Atualizar layout
        fig.update_layout(
            title=f"Comparação de Distribuições: {name1} vs {name2}",
            xaxis_title="Valor",
            yaxis_title="Frequência",
            barmode='overlay',
            bargap=0.1,
        )
        
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Erro ao criar histogramas comparativos: {str(e)}")
        st.info("Verifique se os dados são válidos para visualização.")


def display_statistical_header():
    """Exibe header informativo da análise estatística"""
    st.markdown("""
    ### 📊 Análise Estatística Avançada de Ativos
    
    Esta seção implementa **análises estatísticas sofisticadas** para identificar:
    - **Probabilidades de eventos extremos** (quedas > 10%)
    - **Comparação de distribuições** entre ativos
    - **Modelos de risco** com distribuições t-Student vs Normal
    - **Análises de normalização** e **pair trading estatístico**
    """)


def extreme_analysis_tab(stat_analyzer, df):
    """Tab 1: Análise de extremos de ativos"""
    st.subheader("🎯 Análise de Extremos de Ativos")
    
    # Chave para lembrar que estamos na aba de análise de extremos
    tab_state_key = "active_tab_extreme_analysis"
    st.session_state[tab_state_key] = True
    
    # Seleção do ativo
    available_assets = [col for col in df.columns if df[col].dtype in ['float64', 'int64']]
    
    if not available_assets:
        st.error("❌ Nenhum ativo numérico encontrado no dataset")
        st.info("💡 Execute a página Home para baixar dados atualizados")
        return
    
    # Configurações
    selected_asset, threshold = _setup_extreme_analysis_config(available_assets)
    
    if st.button(f"📊 Analisar Extremos - {selected_asset}"):
        _execute_extreme_analysis(stat_analyzer, selected_asset, threshold)


def _setup_extreme_analysis_config(available_assets):
    """Configura parâmetros para análise de extremos"""
    col1, col2 = st.columns(2)
    
    with col1:
        selected_asset = st.selectbox(
            "Selecione o ativo para análise:",
            available_assets,
            index=0 if PETR4_SYMBOL not in available_assets else available_assets.index(PETR4_SYMBOL)
        )
    
    with col2:
        threshold = st.slider(
            "Threshold de queda (%):", 
            5.0, 20.0, 10.0, 1.0
        ) / 100
        st.info("💡 Análise baseada em distribuição empírica e t-Student")
    
    return selected_asset, threshold


def _execute_extreme_analysis(stat_analyzer, selected_asset, threshold):
    """Executa análise de extremos usando o módulo especializado"""
    # Importar o módulo de análise de extremos
    try:
        from extreme_analysis import execute_extreme_analysis
        # Usar a função do módulo especializado
        execute_extreme_analysis(stat_analyzer, selected_asset, threshold)
    except ImportError:
        # Fallback para a implementação anterior se o módulo não estiver disponível
        with st.spinner("Analisando distribuições e extremos..."):
            try:
                # Verificar se o método extreme_analysis_any_asset existe
                if hasattr(stat_analyzer, 'extreme_analysis_any_asset'):
                    extreme_analysis = stat_analyzer.extreme_analysis_any_asset(
                        asset_symbol=selected_asset, 
                        threshold=threshold
                    )
                else:
                    # Cria resultado similar ao esperado
                    asset_returns = stat_analyzer.returns[selected_asset].dropna()
                    extreme_falls = asset_returns[asset_returns <= -threshold]
                    extreme_analysis = {
                        'total_days': len(asset_returns),
                        'extreme_falls_count': len(extreme_falls),
                        'probability': len(extreme_falls) / len(asset_returns),
                        'daily_statistics': {
                            'mean': asset_returns.mean(),
                            'std': asset_returns.std(),
                            'skewness': 0,
                            'kurtosis': 0,
                            'min': asset_returns.min(),
                            'max': asset_returns.max()
                        }
                    }
                
                if 'error' not in extreme_analysis:
                    _display_extreme_analysis_results(extreme_analysis, selected_asset, threshold)
                else:
                    st.error(extreme_analysis['error'])
            except Exception as e:
                st.error(f"Erro na análise: {str(e)}")
                import traceback
                st.error(f"Detalhes: {traceback.format_exc()}")


def _display_extreme_analysis_results(extreme_analysis, selected_asset, threshold):
    """Exibe resultados da análise de extremos"""
    # Métricas principais
    st.subheader(f"📈 Métricas de Risco - {selected_asset}")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        prob_empirical = extreme_analysis.get('probability', 0)
        st.metric(
            f"Prob. Queda > {threshold:.0%}", 
            f"{prob_empirical:.2%}"
        )
    with col2:
        total_days = extreme_analysis.get('total_days', 0)
        extreme_count = extreme_analysis.get('extreme_falls_count', 0)
        st.metric("Eventos Extremos", f"{extreme_count}/{total_days}")
    with col3:
        daily_stats = extreme_analysis.get('daily_statistics', {})
        daily_vol = daily_stats.get('std', 0)
        annual_vol = daily_vol * np.sqrt(252)
        st.metric("Volatilidade Anual", f"{annual_vol:.1%}")
    with col4:
        skewness = daily_stats.get('skewness', 0)
        st.metric("Assimetria", f"{skewness:.2f}")
    
    # Interpretação dos resultados
    _display_extreme_interpretation(prob_empirical, threshold)


def _display_extreme_interpretation(prob_empirical, threshold):
    """Exibe interpretação dos resultados de extremos"""
    st.subheader("🎯 Interpretação dos Resultados")
    
    if prob_empirical > 0.05:  # 5%
        st.warning(f"""
        ⚠️ **Alto Risco**: Probabilidade de {prob_empirical:.1%} para quedas superiores a {threshold:.0%} 
        indica volatilidade elevada. Considere estratégias de hedge.
        """)
    elif prob_empirical > 0.02:  # 2%
        st.info(f"""
        💡 **Risco Moderado**: Probabilidade de {prob_empirical:.1%} é significativa. 
        Monitore indicadores macro e setoriais.
        """)
    else:
        st.success(f"""
        ✅ **Risco Baixo**: Probabilidade de {prob_empirical:.1%} é relativamente baixa 
        para quedas extremas no horizonte analisado.
        """)


def distribution_comparison_tab(stat_analyzer):
    """Tab 2: Comparação de distribuições"""
    st.subheader("📈 Comparação Estatística de Distribuições")
    
    # Chave para lembrar que estamos na aba de comparação de distribuições
    tab_state_key = "active_tab_distribution_comparison"
    st.session_state[tab_state_key] = True
    
    # Configurações
    st.markdown("### ⚙️ Configurações da Análise")
    
    # Opções: busca automática ou seleção manual de ativos
    comparison_option = st.radio(
        "Escolha o método de comparação:",
        ["🔍 Busca automática de pares diferentes", "👆 Selecionar ativos manualmente"],
        horizontal=True
    )
    
    if comparison_option == "🔍 Busca automática de pares diferentes":
        # Configurações para busca automática
        min_observations = _setup_distribution_comparison_config()
        
        if st.button("🔍 Encontrar Distribuições Diferentes"):
            _execute_distribution_comparison(stat_analyzer, min_observations)
    else:
        # Seleção manual de dois ativos
        available_assets = sorted(stat_analyzer.returns.columns.tolist())
        
        col1, col2 = st.columns(2)
        with col1:
            asset1 = st.selectbox("Selecione o primeiro ativo:", available_assets, index=0)
        
        # Filtrando o segundo ativo para não ser o mesmo que o primeiro
        filtered_assets = [asset for asset in available_assets if asset != asset1]
        
        with col2:
            asset2 = st.selectbox("Selecione o segundo ativo:", filtered_assets, index=0)
            
        min_observations = st.slider("Mínimo de observações:", 100, 1000, 252)
            
        if st.button("📊 Comparar Distribuições"):
            _execute_custom_distribution_comparison(stat_analyzer, asset1, asset2, min_observations)


def _setup_distribution_comparison_config():
    """Configura parâmetros para comparação de distribuições"""
    col1, col2 = st.columns(2)
    
    with col1:
        min_observations = st.slider("Mínimo de observações:", 100, 1000, 252)
    with col2:
        st.info("💡 Análise baseada em testes de Kolmogorov-Smirnov e Mann-Whitney U")
    
    return min_observations


def _execute_distribution_comparison(stat_analyzer, min_observations):
    """Executa comparação de distribuições"""
    with st.spinner("Comparando distribuições entre ativos..."):
        try:
            different_pairs_result = stat_analyzer.find_different_distributions(
                min_data_points=min_observations
            )
            
            if different_pairs_result and isinstance(different_pairs_result, dict) and 'error' not in different_pairs_result:
                _display_distribution_comparison_results(stat_analyzer, different_pairs_result)
            else:
                error_msg = different_pairs_result.get('error', 'Nenhum par encontrado') if isinstance(different_pairs_result, dict) else 'Nenhum par encontrado'
                st.info(f"📊 {error_msg}")
                st.info("💡 Tente ajustar o mínimo de observações ou verificar se há dados suficientes.")
        except Exception as e:
            st.error(f"❌ Erro ao comparar distribuições: {str(e)}")
            st.info("💡 Tente ajustar o mínimo de observações ou verificar se há dados suficientes.")


def _execute_custom_distribution_comparison(stat_analyzer, asset1, asset2, min_observations):
    """Executa comparação de distribuições para dois ativos específicos selecionados manualmente"""
    with st.spinner(f"Comparando distribuições entre {asset1} e {asset2}..."):
        try:
            # Extrair retornos dos dois ativos
            returns1 = stat_analyzer.returns[asset1].dropna() if asset1 in stat_analyzer.returns.columns else pd.Series()
            returns2 = stat_analyzer.returns[asset2].dropna() if asset2 in stat_analyzer.returns.columns else pd.Series()
            
            # Verificar se os ativos existem nos dados
            if returns1.empty or returns2.empty:
                st.error(f"❌ Um ou ambos os ativos não foram encontrados nos dados")
                return
            
            # Verificar se há dados suficientes
            if len(returns1) < min_observations or len(returns2) < min_observations:
                st.warning(f"⚠️ Um ou ambos os ativos têm menos que {min_observations} observações")
                st.info("💡 Os resultados podem não ser estatisticamente significativos")
              # Alinhar as séries temporais
            # Verificar intersecção de índices
            common_index = returns1.index.intersection(returns2.index)
            if len(common_index) < 10:
                st.error(f"❌ Dados alinhados insuficientes para análise estatística")
                st.info(f"Encontrados apenas {len(common_index)} pontos de dados comuns entre os ativos.")
                return
            
            # Filtrar apenas pontos de dados em comum
            aligned_returns1 = returns1.loc[common_index]
            aligned_returns2 = returns2.loc[common_index]
            
            # Verificar se ainda há dados após filtrar valores NaN
            aligned_data = pd.concat([aligned_returns1, aligned_returns2], axis=1, keys=[asset1, asset2]).dropna()
            
            if aligned_data.empty or aligned_data.shape[0] < 10:
                st.error(f"❌ Dados alinhados insuficientes para análise estatística")
                st.info(f"Encontrados apenas {aligned_data.shape[0]} pontos de dados válidos após remover valores NaN.")
                return
                
            # Usar os dados já alinhados
            aligned_returns1 = aligned_data[asset1]
            aligned_returns2 = aligned_data[asset2]
            
            # Verificação adicional para valores extremos ou inválidos
            if (aligned_returns1.abs() > 1).any() or (aligned_returns2.abs() > 1).any():
                st.warning("⚠️ Os dados contêm valores extremos (retornos superiores a 100%). Verifique os dados de entrada.")
                # Podemos continuar, mas o usuário foi avisado
            
            # Realizar testes estatísticos
            # Teste de Kolmogorov-Smirnov
            ks_statistic, ks_pvalue = stats.ks_2samp(aligned_returns1, aligned_returns2)
            
            # Teste de Mann-Whitney U
            try:
                mw_statistic, mw_pvalue = stats.mannwhitneyu(aligned_returns1, aligned_returns2, alternative='two-sided')
            except ValueError as e:
                st.warning(f"Não foi possível realizar o teste Mann-Whitney U: {str(e)}")
                mw_statistic, mw_pvalue = 0, 1.0  # Valores padrão indicando sem diferença significativa
            
            # Teste de Levene para igualdade de variâncias
            try:
                levene_statistic, levene_pvalue = stats.levene(aligned_returns1, aligned_returns2)
            except ValueError as e:
                st.warning(f"Não foi possível realizar o teste Levene: {str(e)}")
                levene_statistic, levene_pvalue = 0, 1.0  # Valores padrão indicando sem diferença significativa
              # Criar dicionário de resultados
            comparison_tests = {
                'ks_test': {
                    'statistic': ks_statistic,
                    'p_value': ks_pvalue,
                    'significant': ks_pvalue < 0.05
                },
                'mann_whitney': {
                    'statistic': mw_statistic,
                    'p_value': mw_pvalue,
                    'significant': mw_pvalue < 0.05
                },
                'levene': {
                    'statistic': levene_statistic,
                    'p_value': levene_pvalue,
                    'significant': levene_pvalue < 0.05
                }
            }
            
            # Calcular estatísticas descritivas usando as séries alinhadas
            try:
                skew1 = skew(aligned_returns1) if len(aligned_returns1) > 3 else 0
                kurt1 = kurtosis(aligned_returns1) if len(aligned_returns1) > 4 else 0
                skew2 = skew(aligned_returns2) if len(aligned_returns2) > 3 else 0
                kurt2 = kurtosis(aligned_returns2) if len(aligned_returns2) > 4 else 0
            except Exception as e:
                st.warning(f"Erro ao calcular estatísticas de distribuição: {str(e)}")
                skew1, kurt1, skew2, kurt2 = 0, 0, 0, 0
            
            descriptive_stats = {
                asset1: {
                    'mean': aligned_returns1.mean(),
                    'median': aligned_returns1.median(),
                    'std': aligned_returns1.std(),
                    'skew': skew1,
                    'kurtosis': kurt1
                },
                asset2: {
                    'mean': aligned_returns2.mean(),
                    'median': aligned_returns2.median(),
                    'std': aligned_returns2.std(),
                    'skew': skew2,
                    'kurtosis': kurt2
                }
            }
            
            # Resultado da comparação
            result = {
                'assets': {
                    'asset1': asset1,
                    'asset2': asset2
                },
                'comparison_tests': comparison_tests,
                'descriptive_stats': descriptive_stats,
                'data_points': {
                    asset1: len(returns1),
                    asset2: len(returns2)
                }
            }
            
            # Exibir resultados da comparação
            _display_manual_comparison_results(stat_analyzer, result)
            
        except Exception as e:
            st.error(f"❌ Erro ao comparar distribuições: {str(e)}")
            st.info("💡 Verifique se os ativos selecionados têm dados suficientes e válidos.")


def _display_distribution_comparison_results(stat_analyzer, different_pairs_result):
    """Exibe resultados da comparação de distribuições"""
    st.success("🎯 Encontrado par com distribuições estatisticamente diferentes!")
    
    # Exibir resultados da comparação
    st.subheader("📊 Análise de Distribuições Diferentes")
    
    assets = different_pairs_result['assets']
    st.write(f"**Ativos analisados:** {assets['asset1']} vs {assets['asset2']}")
    
    # Testes estatísticos
    _display_statistical_tests(different_pairs_result)
    
    # Estatísticas descritivas
    _display_descriptive_statistics(different_pairs_result, assets)
    
    # Criar gráfico de comparação
    comparison_plot = stat_analyzer.create_distribution_comparison_plot(
        assets['asset1'], assets['asset2']
    )
    if comparison_plot:
        st.plotly_chart(comparison_plot, use_container_width=True)
    
    # Adicionar botão de download PDF
    from pdf_export_helpers import add_download_button_to_distribution_comparison
    
    # Extrair dados necessários para o PDF
    asset1 = assets['asset1']
    asset2 = assets['asset2']
    comparison_tests = different_pairs_result.get('comparison_tests', {})
    descriptive_stats = different_pairs_result.get('descriptive_stats', {})
    data_points = different_pairs_result.get('data_points', {})
    
    add_download_button_to_distribution_comparison(
        asset1, asset2, comparison_tests, descriptive_stats, data_points
    )


def _display_manual_comparison_results(stat_analyzer, result):
    """Exibe resultados da comparação manual de distribuições"""
    assets = result['assets']
    asset1 = assets['asset1']
    asset2 = assets['asset2']
    
    # Header com resultado principal
    are_different = False
    if 'comparison_tests' in result:
        ks_significant = result['comparison_tests'].get('ks_test', {}).get('significant', False)
        mw_significant = result['comparison_tests'].get('mann_whitney', {}).get('significant', False)
        are_different = ks_significant or mw_significant
    
    if are_different:
        st.success(f"✅ Os ativos {asset1} e {asset2} têm distribuições estatisticamente diferentes!")
    else:
        st.info(f"ℹ️ Não foi encontrada diferença significativa entre as distribuições de {asset1} e {asset2}.")
    
    # Informação sobre os dados
    data_points = result.get('data_points', {})
    st.write(f"**Observações:** {asset1}: {data_points.get(asset1, 'N/A')}, {asset2}: {data_points.get(asset2, 'N/A')}")
    
    # Testes estatísticos
    _display_statistical_tests(result)
    
    # Estatísticas descritivas
    _display_descriptive_statistics(result, assets)
    
    # Visualizações
    st.subheader("📊 Visualizações Comparativas")
    
    # Obter os dados de retornos
    returns1 = stat_analyzer.returns[asset1].dropna()
    returns2 = stat_analyzer.returns[asset2].dropna()
    
    # Histograma de comparação
    plot_histogram_comparison(returns1, returns2, asset1, asset2)
    
    # Box plot
    plot_box_comparison(returns1, returns2, asset1, asset2)
    
    # QQ Plot
    plot_qq_comparison(returns1, returns2, asset1, asset2)
    
    # Dispersão
    plot_scatter_chart(returns1, returns2, asset1, asset2)
      # Interpretação dos resultados
    st.subheader("💡 Interpretação dos Resultados")
    
    if are_different:
        st.markdown(f"""
        **Conclusão:** As distribuições de retornos de {asset1} e {asset2} são estatisticamente diferentes, 
        o que indica comportamentos de mercado distintos. Isto pode ser relevante para:
        
        - **Diversificação de portfolio**: ativos com comportamentos diferentes ajudam na diversificação
        - **Pair trading**: confirme também a cointegração para estratégias de pair trading
        - **Alocação de risco**: o ativo com maior volatilidade deve receber menor alocação para balanceamento
        """)
    else:
        st.markdown(f"""
        **Conclusão:** As distribuições de {asset1} e {asset2} não apresentaram diferença estatisticamente 
        significativa, o que sugere comportamentos similares em termos de distribuição de retornos.
        
        - **Diversificação limitada**: estes ativos podem oferecer menos benefícios de diversificação
        - **Correlação**: verifique a correlação entre eles para entender se movem juntos
        - **Análise setorial**: podem pertencer ao mesmo setor ou ser afetados pelos mesmos fatores
        """)
        
    # Adicionar botão de download PDF
    from pdf_export_helpers import add_download_button_to_distribution_comparison
    
    # Extrair dados necessários para o PDF
    comparison_tests = result.get('comparison_tests', {})
    descriptive_stats = result.get('descriptive_stats', {})
    
    add_download_button_to_distribution_comparison(
        asset1, asset2, comparison_tests, descriptive_stats, data_points
    )


def _display_statistical_tests(different_pairs_result):
    """Exibe testes estatísticos"""
    comparison_tests = different_pairs_result.get('comparison_tests', {})
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🧪 Testes Estatísticos")
        
        # Kolmogorov-Smirnov
        if 'ks_test' in comparison_tests:
            ks_test = comparison_tests['ks_test']
            ks_status = "✅ Significativo" if ks_test.get('significant', False) else "❌ Não significativo"
            st.metric("Teste K-S", f"p = {ks_test.get('p_value', 0):.4f}", ks_status)
        else:
            st.metric("Teste K-S", "Não disponível")
        
        # Mann-Whitney U
        if 'mann_whitney' in comparison_tests:
            mw_test = comparison_tests['mann_whitney']
            mw_status = "✅ Significativo" if mw_test.get('significant', False) else "❌ Não significativo"
            st.metric("Mann-Whitney U", f"p = {mw_test.get('p_value', 0):.4f}", mw_status)
        else:
            st.metric("Mann-Whitney U", "Não disponível")
    
    with col2:
        st.subheader("📊 Métricas de Comparação")
        
        # KS test
        if 'ks_test' in comparison_tests:
            st.metric(
                "Kolmogorov-Smirnov Test (p-value)",
                f"{comparison_tests['ks_test'].get('p_value', 'N/A'):.4f}",
                help="Testa se duas amostras são da mesma distribuição. p-value < 0.05 indica distribuições diferentes."
            )
        
        # Anderson-Darling test
        if 'anderson_darling' in comparison_tests:
            st.metric(
                "Anderson-Darling Test",
                f"{comparison_tests['anderson_darling'].get('statistic', 'N/A'):.4f}",
                help="Testa a normalidade. Valores maiores indicam maior desvio da normalidade."
            )
        
        # Mann-Whitney test
        if 'mann_whitney' in comparison_tests:
            st.metric(
                "Mann-Whitney U Test (p-value)",
                f"{comparison_tests['mann_whitney'].get('p_value', 'N/A'):.4f}",
                help="Testa se as medianas são iguais. p-value < 0.05 indica medianas diferentes."
            )
        
        # Levene test
        if 'levene' in comparison_tests:
            st.metric(
                "Levene Test (p-value)",
                f"{comparison_tests['levene'].get('p_value', 'N/A'):.4f}",
                help="Testa se as variâncias são iguais. p-value < 0.05 indica variâncias diferentes."
            )


def _display_descriptive_statistics(different_pairs_result, assets):
    """Exibe estatísticas descritivas dos ativos comparados"""
    if 'descriptive_stats' not in different_pairs_result:
        st.warning("⚠️ Estatísticas descritivas não calculadas para este par.")
        return
    
    stats = different_pairs_result['descriptive_stats']
    
    st.subheader("📈 Estatísticas Descritivas")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**{assets['asset1']}**")
        if assets['asset1'] in stats:
            asset1_stats = stats[assets['asset1']]
            stats_df1 = pd.DataFrame({
                "Métrica": ["Média", "Mediana", "Desvio Padrão", "Assimetria", "Curtose"],
                "Valor": [
                    f"{asset1_stats.get('mean', 'N/A'):.5f}",
                    f"{asset1_stats.get('median', 'N/A'):.5f}",
                    f"{asset1_stats.get('std', 'N/A'):.5f}",
                    f"{asset1_stats.get('skew', 'N/A'):.5f}",
                    f"{asset1_stats.get('kurtosis', 'N/A'):.5f}"
                ]
            })
            st.dataframe(stats_df1, hide_index=True)
        else:
            st.info(f"Não há estatísticas disponíveis para {assets['asset1']}")
    
    with col2:
        st.write(f"**{assets['asset2']}**")
        if assets['asset2'] in stats:
            asset2_stats = stats[assets['asset2']]
            stats_df2 = pd.DataFrame({
                "Métrica": ["Média", "Mediana", "Desvio Padrão", "Assimetria", "Curtose"],
                "Valor": [
                    f"{asset2_stats.get('mean', 'N/A'):.5f}",
                    f"{asset2_stats.get('median', 'N/A'):.5f}",
                    f"{asset2_stats.get('std', 'N/A'):.5f}",
                    f"{asset2_stats.get('skew', 'N/A'):.5f}",
                    f"{asset2_stats.get('kurtosis', 'N/A'):.5f}"
                ]
            })
            st.dataframe(stats_df2, hide_index=True)
        else:
            st.info(f"Não há estatísticas disponíveis para {assets['asset2']}")


def plot_correlation_heatmap(returns, selected_assets=None, method='pearson'):
    """
    Plota um mapa de calor de correlações entre ativos selecionados.
    
    Parâmetros:
    -----------
    returns : DataFrame
        DataFrame com retornos dos ativos
    selected_assets : list, opcional
        Lista de ativos para incluir no mapa. Se None, usa todos os ativos.
    method : str, default='pearson'
        Método de correlação ('pearson', 'spearman', ou 'kendall')
        
    Retorno:
    --------
    None
    """
    # Filtrar ativos selecionados
    if selected_assets and len(selected_assets) > 0:
        filtered_returns = returns[selected_assets].copy()
    else:
        filtered_returns = returns.copy()
    
    # Verificar se há ativos suficientes
    if filtered_returns.shape[1] < 2:
        st.warning("Selecione pelo menos 2 ativos para o mapa de correlação.")
        return
    
    # Calcular matriz de correlação
    corr_matrix = filtered_returns.corr(method=method)
    
    # Criar mapa de calor
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        color_continuous_scale='RdBu_r',
        title=f"Mapa de Correlação ({method.capitalize()})",
        aspect="auto",
        labels=dict(x="Ativo", y="Ativo", color="Correlação")
    )
    
    # Ajustar layout
    fig.update_layout(height=600)
    
    st.plotly_chart(fig, use_container_width=True)


def plot_box_comparison(data1, data2, name1, name2):
    """
    Plota boxplots comparativos de duas séries de dados.
    
    Parâmetros:
    -----------
    data1 : array ou Series
        Primeira série de dados
    data2 : array ou Series
        Segunda série de dados
    name1 : str
        Nome da primeira série
    name2 : str
        Nome da segunda série
        
    Retorno:
    --------
    None
    """
    # Criar DataFrame para plotly
    box_df = pd.DataFrame({
        'Valor': pd.concat([pd.Series(data1), pd.Series(data2)]),
        'Ativo': [name1] * len(data1) + [name2] * len(data2)
    })
    
    # Criar boxplot comparativo
    fig = px.box(
        box_df, 
        x='Ativo', 
        y='Valor', 
        title=f"Comparação de Boxplots: {name1} vs {name2}",
        color='Ativo'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_qq_comparison(data1, data2, name1, name2):
    """
    Plota gráficos QQ (quantil-quantil) para duas séries de dados.
    
    Parâmetros:
    -----------
    data1 : array ou Series
        Primeira série de dados
    data2 : array ou Series
        Segunda série de dados
    name1 : str
        Nome da primeira série
    name2 : str
        Nome da segunda série
        
    Retorno:
    --------
    None
    """
    # Verificação inicial de dados
    if isinstance(data1, (list, np.ndarray)) and len(data1) == 0:
        st.warning(f"Não há dados disponíveis para {name1}")
        return
    
    if isinstance(data2, (list, np.ndarray)) and len(data2) == 0:
        st.warning(f"Não há dados disponíveis para {name2}")
        return
    
    # Criar duas subplots
    fig = make_subplots(rows=1, cols=2, 
                        subplot_titles=[f"QQ Plot - {name1}", f"QQ Plot - {name2}"])
      # Calcular quantis teóricos (distribuição normal)
    from scipy import stats    # QQ plot para data1
    data1_clean = pd.Series(data1).dropna()
    if len(data1_clean) > 2:  # Necessário pelo menos 3 pontos
        try:
            # Verificar se há valores válidos suficientes após limpeza
            if data1_clean.isnull().all() or len(data1_clean) == 0:
                st.warning(f"Não há dados válidos para {name1} após remover valores NaN")
                has_plot1 = False
            else:
                qq_plot_asset1 = stats.probplot(data1_clean, dist="norm", fit=True)
                theoretical_quantiles_asset1, ordered_values_asset1 = qq_plot_asset1[0]
                (regression_slope_asset1, regression_intercept_asset1, correlation_coef_asset1) = qq_plot_asset1[1]  # Corrigido para atribuir todos os valores retornados
                
                fig.add_trace(
                    go.Scatter(
                        x=theoretical_quantiles_asset1,
                        y=ordered_values_asset1,
                        mode='markers',
                        name=name1,
                        marker=dict(color='blue')
                    ),
                    row=1, col=1
                )
                
                # Adicionar linha de referência
                fig.add_trace(
                    go.Scatter(
                        x=theoretical_quantiles_asset1,
                        y=regression_intercept_asset1 + regression_slope_asset1 * theoretical_quantiles_asset1,
                        mode='lines',
                        name='Referência Normal',
                        line=dict(color='red')
                    ),
                    row=1, col=1
                )
                has_plot1 = True
        except Exception as e:
            st.warning(f"Erro ao plotar QQ para {name1}: {str(e)}")
            has_plot1 = False
    else:
        st.warning(f"Dados insuficientes para {name1}. Necessário pelo menos 3 pontos.")
        has_plot1 = False
              # QQ plot para data2
    data2_clean = pd.Series(data2).dropna()
    if len(data2_clean) > 2:  # Necessário pelo menos 3 pontos
        try:
            # Verificar se há valores válidos suficientes após limpeza
            if data2_clean.isnull().all() or len(data2_clean) == 0:
                st.warning(f"Não há dados válidos para {name2} após remover valores NaN")
                has_plot2 = False
            else:
                qq_plot_asset2 = stats.probplot(data2_clean, dist="norm", fit=True)
                theoretical_quantiles_asset2, ordered_values_asset2 = qq_plot_asset2[0]
                (regression_slope_asset2, regression_intercept_asset2, correlation_coef_asset2) = qq_plot_asset2[1]  # Corrigido para atribuir todos os valores retornados
                
                fig.add_trace(
                    go.Scatter(
                        x=theoretical_quantiles_asset2,
                        y=ordered_values_asset2,
                        mode='markers',
                        name=name2,
                        marker=dict(color='green')
                    ),
                    row=1, col=2
                )
                
                # Adicionar linha de referência
                fig.add_trace(
                    go.Scatter(
                        x=theoretical_quantiles_asset2,
                        y=regression_intercept_asset2 + regression_slope_asset2 * theoretical_quantiles_asset2,
                        mode='lines',
                        name='Referência Normal',
                        line=dict(color='red')
                    ),
                    row=1, col=2
                )
                has_plot2 = True
        except Exception as e:
            st.warning(f"Erro ao plotar QQ para {name2}: {str(e)}")
            has_plot2 = False
    else:
        st.warning(f"Dados insuficientes para {name2}. Necessário pelo menos 3 pontos.")
        has_plot2 = False    # Verificar se pelo menos um gráfico foi criado
    if has_plot1 or has_plot2:
        # Atualizar layout
        fig.update_layout(
            title="Gráficos QQ (Teste de Normalidade)",
            height=500,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("⚠️ Não foi possível criar nenhum gráfico QQ com os dados fornecidos.")

def display_distribution_comparison_metrics(comparison_tests):
    """
    Exibe métricas de comparação entre duas distribuições.
    
    Parâmetros:
    -----------
    comparison_tests : dict
        Dicionário com resultados dos testes estatísticos
        
    Retorno:
    --------
    None
    """
    if not comparison_tests:
        st.warning("Não há dados suficientes para comparação estatística.")
        return
    
    st.subheader("📊 Métricas de Comparação")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # KS test
        if 'ks_test' in comparison_tests:
            st.metric(
                "Kolmogorov-Smirnov Test (p-value)",
                f"{comparison_tests['ks_test'].get('p_value', 'N/A'):.4f}",
                help="Testa se duas amostras são da mesma distribuição. p-value < 0.05 indica distribuições diferentes."
            )
        
        # Anderson-Darling test
        if 'anderson_darling' in comparison_tests:
            st.metric(
                "Anderson-Darling Test",
                f"{comparison_tests['anderson_darling'].get('statistic', 'N/A'):.4f}",
                help="Testa a normalidade. Valores maiores indicam maior desvio da normalidade."
            )
    
    with col2:
        # Mann-Whitney test
        if 'mann_whitney' in comparison_tests:
            st.metric(
                "Mann-Whitney U Test (p-value)",
                f"{comparison_tests['mann_whitney'].get('p_value', 'N/A'):.4f}",
                help="Testa se as medianas são iguais. p-value < 0.05 indica medianas diferentes."
            )
        
        # Levene test
        if 'levene' in comparison_tests:
            st.metric(
                "Levene Test (p-value)",
                f"{comparison_tests['levene'].get('p_value', 'N/A'):.4f}",
                help="Testa se as variâncias são iguais. p-value < 0.05 indica variâncias diferentes."
            )


def risk_models_tab(stat_analyzer, df):
    """
    Tab 3: Modelos de Risco
    
    Implementação de modelos de risco financeiro, incluindo análises de distribuição de retornos
    com suporte para distribuições t-Student e metodologia avançada para eventos extremos.
    
    Inclui análise especial para Petrobras com a implementação da metodologia
    de eventos extremos desenvolvida pelo Prof. Carlos Alberto Rodrigues (UEFS).
    """
    st.subheader("🔬 Modelos de Risco")
    

    
    # Seleção do ativo para análise de risco
    available_assets = [col for col in df.columns if df[col].dtype in ['float64', 'int64']]
    
    if not available_assets:
        st.error("❌ Nenhum ativo numérico disponível para análise")
        return
    
    # Configurações da análise
    col1, col2 = st.columns(2)
    
    with col1:
        selected_asset = st.selectbox(
            "Selecione o ativo para análise de risco:",
            available_assets
        )
    
    with col2:
        confidence_level = st.slider(
            "Nível de confiança (%)",
            min_value=90.0,
            max_value=99.0,
            value=95.0,
            step=0.5,
            help="Nível de confiança para cálculos de risco (VaR, CVaR)"
        )
      # Configuração de análise
    show_petr4_analysis = False
    
    # Análise de Modelos de Risco
    if st.button("🔎 Analisar Risco"):
        _execute_risk_analysis(stat_analyzer, df, selected_asset, confidence_level, show_petr4_analysis)


def _execute_risk_analysis(stat_analyzer, df, selected_asset, confidence_level, show_petr4_analysis):
    """Executa análise de risco"""
    with st.spinner(f"Analisando risco do ativo {selected_asset}..."):
        try:
            # Extrair retornos
            returns = df[selected_asset].pct_change().dropna()
            
            # Estatísticas básicas
            mean_return = returns.mean()
            std_return = returns.std()
            skewness = returns.skew()
            kurt = returns.kurtosis()
            
            # Cálculo de VaR e CVaR
            alpha = 1 - (confidence_level / 100.0)
            var_normal = mean_return + std_return * stats.norm.ppf(alpha)
            var_empirical = returns.quantile(alpha)
            
            # CVaR (Expected Shortfall)
            cvar_empirical = returns[returns <= var_empirical].mean()
            
            # Calcular máximo drawdown
            equity_curve = (1 + returns).cumprod()
            running_max = equity_curve.expanding().max()
            drawdown = (equity_curve / running_max - 1)
            max_drawdown = drawdown.min()
            
            # VaR e CVaR usando distribuição t de Student
            try:
                t_params = stats.t.fit(returns)
                df_param, loc_param, scale_param = t_params
                var_t = loc_param + scale_param * stats.t.ppf(alpha, df_param)
            except Exception:
                var_t = var_normal  # Fallback para o VaR normal
            
            # Exibir resultados
            risk_metrics = {
                'mean_return': mean_return,
                'std_return': std_return,
                'skewness': skewness,
                'kurtosis': kurt,
                'var_normal': var_normal,
                'var_empirical': var_empirical,
                'var_t': var_t,
                'cvar_empirical': cvar_empirical,
                'confidence_level': confidence_level,
                'max_drawdown': max_drawdown
            }
            
            _display_risk_analysis_results(selected_asset, risk_metrics)
            
            # Visualizações
            _display_distribution_comparison_plots(returns, selected_asset)
            # Preparar dados para interpretação de risco
            risk_metrics_for_interp = {
                'var_cvar': {
                    'var_95': var_empirical,
                    'var_99': returns.quantile(0.01),  # VaR a 99%
                    'cvar_95': cvar_empirical,
                    'cvar_99': returns[returns <= returns.quantile(0.01)].mean()  # CVaR a 99%
                },
                'volatility_analysis': {
                    'current_vol_annual': std_return * np.sqrt(252)
                },
                'risk_metrics': {
                    'max_drawdown': max_drawdown
                }
            }  
              # Exibir interpretação de risco usando a função importada de risk_utils.py
            _display_risk_interpretation(risk_metrics_for_interp)
            
        except Exception as e:
            st.error(f"Erro na análise: {str(e)}")
            import traceback
            st.error(f"Detalhe: {traceback.format_exc()}")


def _display_risk_analysis_results(selected_asset, risk_metrics):
    """Exibe resultados da análise de risco"""
    st.subheader(f"📊 Métricas de Risco - {selected_asset}")
    
    # Formatar métricas para exibição
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "VaR Empírico", 
            f"{risk_metrics['var_empirical']*100:.2f}%",
            help=f"Value at Risk empírico (nível de confiança: {risk_metrics['confidence_level']}%)"
        )
    
    with col2:
        st.metric(
            "CVaR Empírico", 
            f"{risk_metrics['cvar_empirical']*100:.2f}%",
            help="Conditional Value at Risk (Expected Shortfall)"
        )
    
    with col3:
        st.metric(
            "VaR Normal", 
            f"{risk_metrics['var_normal']*100:.2f}%",
            help="Value at Risk assumindo distribuição Normal"
        )
    
    with col4:
        st.metric(
            "VaR t-Student", 
            f"{risk_metrics['var_t']*100:.2f}%",
            help="Value at Risk usando distribuição t de Student"
        )
    
    # Segunda linha de métricas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Volatilidade Anual", f"{risk_metrics['std_return'] * np.sqrt(252) * 100:.1f}%")
    
    with col2:
        st.metric("Retorno Médio", f"{risk_metrics['mean_return']*100:.2f}%")
    
    with col3:
        # Interpretação da assimetria
        skew_desc = "Simétrica"
        if risk_metrics['skewness'] > 0.5:
            skew_desc = "Ass. Positiva"
        elif risk_metrics['skewness'] < -0.5:
            skew_desc = "Ass. Negativa"
            
        st.metric("Assimetria", f"{risk_metrics['skewness']:.2f}", skew_desc)
    
    with col4:
        # Interpretação da curtose
        kurt_desc = "Normal"
        if risk_metrics['kurtosis'] > 1:
            kurt_desc = "Caudas Pesadas"
        elif risk_metrics['kurtosis'] < -1:
            kurt_desc = "Caudas Leves"
            
        st.metric("Curtose", f"{risk_metrics['kurtosis']:.2f}", kurt_desc)


def _display_distribution_comparison_plots(returns, selected_asset):
    """Cria gráficos comparativos de distribuições de retornos normais vs empíricas"""
    st.subheader("📈 Comparação de Distribuições")
    
    # Criar distribuições teóricas para comparação
    mean = returns.mean()
    std = returns.std()
    
    # Valores para plotagem
    x_range = np.linspace(returns.min(), returns.max(), 100)
    pdf_normal = stats.norm.pdf(x_range, mean, std)
    
    # Ajustar t-student
    t_params = stats.t.fit(returns)
    df_param, loc_param, scale_param = t_params
    pdf_t = stats.t.pdf(x_range, df_param, loc_param, scale_param)
    
    # Gráfico de distribuição
    fig = go.Figure()
    
    # Histograma dos retornos
    fig.add_trace(go.Histogram(
        x=returns,
        histnorm='probability density',
        name='Retornos Empíricos',
        opacity=0.7,
        nbinsx=30,
    ))
    
    # Curva normal
    fig.add_trace(go.Scatter(
        x=x_range,
        y=pdf_normal,
        mode='lines',
        name='Normal',
        line=dict(color='red')
    ))
    
    # Curva t-student
    fig.add_trace(go.Scatter(
        x=x_range,
        y=pdf_t,
        mode='lines',
        name=f't-Student (df={df_param:.1f})',
        line=dict(color='green')
    ))
    
    # Layout do gráfico
    fig.update_layout(
        title=f"Distribuição de Retornos: {selected_asset}",
        xaxis_title="Retorno",
        yaxis_title="Densidade",
        legend_title="Distribuições",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # QQ Plot
    qq_fig = make_subplots(rows=1, cols=2, subplot_titles=["QQ Plot (Normal)", "QQ Plot (t-Student)"])
      # QQ Plot para distribuição normal
    theoretical_quantiles, ordered_values = stats.probplot(returns, dist='norm', fit=False)
    
    qq_fig.add_trace(
        go.Scatter(
            x=theoretical_quantiles,
            y=ordered_values,
            mode='markers',
            name='Dados Observados',
            marker=dict(
                size=5,
                color='rgba(0, 0, 255, 0.7)',
                line=dict(width=1, color='rgba(0, 0, 255, 1)')
            )
        ),
        row=1, col=1
    )
      # Linha de referência (y=x)
    min_val = min(theoretical_quantiles)
    max_val = max(theoretical_quantiles)
    
    qq_fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Linha de Referência',
            line=dict(color='red', width=2, dash='dash')
        ),
        row=1, col=1
    )
      # QQ Plot para t-student
    t_quantiles = stats.t.ppf(np.linspace(0.01, 0.99, len(returns)), df_param)
    t_quantiles = (t_quantiles - t_quantiles.mean()) / t_quantiles.std() * returns.std() + returns.mean()
    
    qq_fig.add_trace(
        go.Scatter(
            x=t_quantiles,
            y=np.sort(returns),
            mode='markers',
            name='Dados Observados',
            marker=dict(
                size=5,
                color='rgba(0, 100, 80, 0.7)',
                line=dict(width=1, color='rgba(0, 100, 80, 1)')
            )
        ),
        row=1, col=2
    )
      # Linha de referência para t-student
    min_t = min(t_quantiles)
    max_t = max(t_quantiles)
    
    qq_fig.add_trace(
        go.Scatter(
            x=[min_t, max_t],
            y=[min(returns), max(returns)],
            mode='lines',
            name='Linha de Referência',
            line=dict(color='red', width=2, dash='dash')        ),
        row=1, col=2
    )
    
    # Melhorando a aparência visual dos gráficos
    qq_fig.update_layout(
        height=500,  # Aumentamos a altura para melhor visualização
        title_text="QQ Plots - Avaliação da Normalidade",
        title_font=dict(size=18),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_white",  # Tema mais limpo
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    # Adicionando títulos aos subplots
    qq_fig.update_xaxes(title_text="Quantis Teóricos Normal", row=1, col=1)
    qq_fig.update_yaxes(title_text="Quantis Observados", row=1, col=1)
    qq_fig.update_xaxes(title_text="Quantis Teóricos t-Student", row=1, col=2)
    qq_fig.update_yaxes(title_text="Quantis Observados", row=1, col=2)
    
    # Exibindo o gráfico aprimorado
    st.plotly_chart(qq_fig, use_container_width=True)


def advanced_pair_trading_tab(df):
    """
    Tab 4: Pair Trading Avançado
    
    Implementa análise estatística avançada para pares de trading,
    incluindo testes de cointegração, análise de resíduos e modelagem
    estatística para estratégias de pair trading.
    """
    st.subheader("🔄 Pair Trading Avançado")
    
    st.markdown("""
    ### 📊 Análise Estatística para Pair Trading
    
    Esta seção implementa análises avançadas para encontrar pares de ativos
    adequados para estratégias de pair trading estatístico.
    """)
    
    # Verificar se há dados suficientes
    if df is None or df.empty or df.shape[1] < 2:
        st.error("❌ Dados insuficientes para análise de pair trading")
        st.info("📈 É necessário ter pelo menos 2 ativos para análise de pares")
        return
    
    # Seleção de ativos
    st.markdown("### ⚙️ Configurações de Análise")
    
    col1, col2 = st.columns(2)
    
    with col1:
        asset1 = st.selectbox(
            "Selecione o primeiro ativo:",
            df.columns,
            key="pair_trading_asset1"
        )
    
    with col2:
        remaining_assets = [asset for asset in df.columns if asset != asset1]
        asset2 = st.selectbox(
            "Selecione o segundo ativo:",
            remaining_assets,
            key="pair_trading_asset2"
        )
    
    # Seleção do período para análise
    start_date = st.date_input(
        "Data inicial para análise:",
        value=df.index[0].date() if not df.empty and len(df.index) > 0 else None,
        min_value=df.index[0].date() if not df.empty and len(df.index) > 0 else None,
        max_value=df.index[-1].date() if not df.empty and len(df.index) > 0 else None
    )
    
    # Botão para executar análise
    if st.button("🔍 Analisar Par"):
        with st.spinner("Realizando análise estatística do par..."):
            try:
                # Extrair dados do par
                pair_data = df[[asset1, asset2]].copy()
                
                # Filtrar por data
                if start_date:
                    pair_data = pair_data[pair_data.index.date >= start_date]
                
                # Calcular retornos
                returns_data = pair_data.pct_change().dropna()
                
                # Verificar se há dados suficientes
                if len(returns_data) < 30:
                    st.error("❌ Dados insuficientes para análise robusta")
                    st.info("📊 Recomendamos pelo menos 30 observações")
                    return
                
                # Exibir análise do par
                _display_pair_analysis(pair_data, returns_data, asset1, asset2)
                
            except Exception as e:
                st.error(f"❌ Erro na análise: {str(e)}")


def _display_pair_analysis(price_data, returns_data, asset1, asset2):
    """Exibe resultados da análise de par trading"""
    st.subheader(f"📊 Análise do Par: {asset1} vs {asset2}")
    
    # Validação abrangente dos dados para preço
    is_valid_price, price_message = validate_data_for_operations(
        price_data, 
        operation_name="análise de preços", 
        min_samples=10, 
        check_columns=[asset1, asset2]
    )
    
    if not is_valid_price:
        st.error(price_message)
        st.info("📊 Verifique se o período selecionado contém dados válidos")
        return
    elif "⚠️" in price_message:
        st.warning(price_message)
    
    # Validação abrangente dos dados para retornos
    is_valid_returns, returns_message = validate_data_for_operations(
        returns_data, 
        operation_name="análise de retornos", 
        min_samples=10, 
        check_columns=[asset1, asset2]
    )
    
    if not is_valid_returns:
        st.error(returns_message)
        st.info("📊 Os cálculos de retornos podem ser afetados por problemas nos dados")
    elif "⚠️" in returns_message:
        st.warning(returns_message)
    
    try:
        # Gráfico de preços normalizados
        st.markdown("### 📈 Evolução de Preços Normalizados")
        
        # Verifica valores ausentes ou inválidos antes da normalização
        if price_data.iloc[0].isna().any() or (price_data.iloc[0] == 0).any():
            st.warning("⚠️ Valores iniciais ausentes ou zero. Usando os primeiros valores válidos para normalização.")
            first_valid = price_data.apply(lambda x: x.first_valid_index())
            normalized_prices = pd.DataFrame(index=price_data.index)
            for col in price_data.columns:
                idx = first_valid[col]
                if idx is not None and price_data.loc[idx, col] != 0:
                    normalized_prices[col] = price_data[col] / price_data.loc[idx, col] * 100
                else:
                    normalized_prices[col] = np.nan
                    st.warning(f"❌ Não foi possível normalizar {col} devido a valores ausentes ou zero")
        else:
            # Normalizar preços
            normalized_prices = price_data.div(price_data.iloc[0]) * 100
        
        # Plotar gráfico de preços normalizados
        fig_prices = px.line(
            normalized_prices, 
            title=f"Preços Normalizados: {asset1} vs {asset2}"
        )
        st.plotly_chart(fig_prices, use_container_width=True)
    except Exception as e:
        st.error(f"❌ Erro ao criar gráfico de preços normalizados: {str(e)}")
        st.info("📊 Verifique se os dados de preço são válidos")
    # Correlação e estatísticas
    st.markdown("### 📊 Métricas Estatísticas")
    
    # Verificar se há dados válidos para cálculos
    valid_data = returns_data.dropna()
    if valid_data.empty or len(valid_data) < 2:
        st.error("❌ Dados insuficientes para calcular métricas estatísticas")
        return
        
    try:
        # Verifica se as colunas existem nos dados
        if asset1 not in valid_data.columns or asset2 not in valid_data.columns:
            st.error(f"❌ Colunas {asset1} ou {asset2} não encontradas nos dados")
            return
            
        # Verificar dados ausentes
        if valid_data[asset1].isna().any() or valid_data[asset2].isna().any():
            st.warning("⚠️ Há valores ausentes nos dados. Removendo para cálculos.")
            valid_data = valid_data.dropna(subset=[asset1, asset2])
            
        # Verificar tamanho dos dados depois da remoção de valores ausentes
        if len(valid_data) < 2:
            st.error("❌ Dados insuficientes após remoção de valores ausentes")
            return
            
        # Calcular correlação com tratamento de erro
        try:
            corr = valid_data[asset1].corr(valid_data[asset2])
            if np.isnan(corr):
                corr = np.nan
                st.warning("⚠️ Não foi possível calcular correlação válida")
        except Exception:
            corr = np.nan
            st.warning("⚠️ Erro ao calcular correlação")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Correlação", f"{corr:.3f}" if not np.isnan(corr) else "N/A")
        
        with col2:
            try:
                # Cálculo do beta com verificações adicionais
                asset2_var = valid_data[asset2].var()
                if asset2_var > 0 and len(valid_data) >= 2:
                    cov_matrix = np.cov(valid_data[asset1], valid_data[asset2])
                    if cov_matrix.shape == (2, 2): # Verificação adicional da matriz de covariância
                        beta = cov_matrix[0, 1] / asset2_var
                        st.metric("Beta", f"{beta:.3f}" if not np.isnan(beta) else "N/A")
                    else:
                        st.metric("Beta", "N/A")
                        st.warning("⚠️ Matriz de covariância inválida")
                else:
                    st.metric("Beta", "N/A")
                    st.warning(f"⚠️ Não foi possível calcular o Beta: variância de {asset2} é zero ou insuficiente")
            except Exception as e:
                st.metric("Beta", "N/A")
                st.warning(f"⚠️ Erro no cálculo do Beta: {str(e)}")
        
        with col3:            
            try:
                # Verificação para cálculo da razão de preços
                if not price_data.empty and price_data[asset2].iloc[-1] != 0:
                    # Verificar valores NaN
                    if not (np.isnan(price_data[asset1].iloc[-1]) or np.isnan(price_data[asset2].iloc[-1])):
                        ratio = price_data[asset1].iloc[-1] / price_data[asset2].iloc[-1]
                        st.metric("Razão de Preços", f"{ratio:.3f}" if not np.isnan(ratio) else "N/A")
                    else:
                        st.metric("Razão de Preços", "N/A")
                        st.warning("⚠️ Valores ausentes nos últimos preços")
                else:
                    st.metric("Razão de Preços", "N/A")
                    st.warning(f"⚠️ Não foi possível calcular a razão de preços: último valor de {asset2} é zero ou ausente")
            except Exception as e:
                st.metric("Razão de Preços", "N/A")
                st.warning(f"⚠️ Erro no cálculo da razão de preços: {str(e)}")
    except Exception as e:
        st.error(f"❌ Erro ao calcular métricas estatísticas: {str(e)}")
        st.info("📊 Verifique se os dados são válidos e contêm informações suficientes")
    
    # Gráficos adicionais
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Dispersão de Retornos")
        plot_scatter_chart(returns_data[asset1], returns_data[asset2], asset1, asset2)
    
    with col2:
        st.markdown("### 📊 Comparação de Distribuições")
        plot_histogram_comparison(returns_data[asset1], returns_data[asset2], asset1, asset2)
    # Testes estatísticos
    st.markdown("### 🧪 Testes Estatísticos")
    
    # Inicializa as variáveis com valores padrão
    coint_pvalue = np.nan
    half_life = np.nan
    adf_pvalue = np.nan
    
    try:
        # Verificação inicial abrangente dos dados
        if price_data.empty:
            st.error("❌ Não há dados disponíveis para testes estatísticos")
            return
            
        # Verificar dados ausentes e zeros (que podem causar problemas)
        missing_data = price_data.isnull().sum()
        zero_values = (price_data == 0).sum()
        
        if missing_data.any() or zero_values.any():
            st.warning(f"⚠️ Dados contêm valores ausentes ({missing_data.sum()}) ou zeros ({zero_values.sum()})")
        
        # Verificar se há dados suficientes para testes estatísticos
        if len(price_data) < 30:
            st.warning(f"⚠️ Dados insuficientes para testes estatísticos robustos (mínimo recomendado: 30, atual: {len(price_data)})")
            return
        
        # Verificar se as colunas existem
        if asset1 not in price_data.columns or asset2 not in price_data.columns:
            st.error(f"❌ Colunas necessárias não encontradas: {asset1} ou {asset2}")
            return
            
        # Remover valores ausentes para evitar problemas em cálculos
        clean_data = price_data[[asset1, asset2]].dropna()
        if len(clean_data) < 30:
            st.warning(f"⚠️ Após remover valores ausentes, restaram apenas {len(clean_data)} observações (mínimo recomendado: 30)")
            return
            
        # Verificar variância zero (dados constantes)
        if clean_data[asset1].var() == 0 or clean_data[asset2].var() == 0:
            st.warning("⚠️ Um ou ambos os ativos têm variância zero (preços constantes)")
            return
            
        # Preparar dados para testes com tratamento de erro
        try:
            X = sm.add_constant(clean_data[asset2])
            model = sm.OLS(clean_data[asset1], X).fit()
        except Exception as e:
            st.error(f"❌ Erro ao criar modelo OLS: {str(e)}")
            return
            
        # Teste de cointegração com validações
        try:
            # Verificar se há dados suficientes e sem valores ausentes
            if len(clean_data[asset1]) > 30 and len(clean_data[asset2]) > 30:
                # Verificar se os dados não são constantes
                if np.std(clean_data[asset1]) > 1e-8 and np.std(clean_data[asset2]) > 1e-8:
                    coint_result = sm.tsa.stattools.coint(clean_data[asset1], clean_data[asset2])
                    coint_pvalue = coint_result[1]
                    
                    # Validar resultado da cointegração
                    if not np.isfinite(coint_pvalue):
                        st.warning("⚠️ Resultado do teste de cointegração é um valor não finito")
                        coint_pvalue = np.nan
                else:
                    st.warning("⚠️ Dados com desvio padrão próximo de zero - teste de cointegração não confiável")
            else:
                st.warning("⚠️ Dados insuficientes para teste de cointegração")
        except Exception as e:
            st.warning(f"⚠️ Erro no teste de cointegração: {str(e)}")
            coint_pvalue = np.nan
        
        # Resíduos para half-life com validações detalhadas
        try:
            if hasattr(model, 'resid') and len(model.resid) > 2:
                resids = model.resid
                
                # Verificar outliers nos resíduos
                z_scores = np.abs(stats.zscore(resids, nan_policy='omit'))
                if np.any(z_scores > 10):  # Outliers extremos
                    st.info("ℹ️ Detectados outliers extremos nos resíduos - considere filtrar os dados")
                
                # Calcular lag de resíduos com tratamento extra para arrays vazios
                lag_resids = pd.Series(resids).shift(1).dropna()
                
                # Só prosseguir se tivermos dados suficientes
                if len(lag_resids) >= 2:
                    # Obter resíduos no mesmo índice dos lags
                    current_resids = pd.Series(resids).iloc[1:len(lag_resids)+1]
                    
                    # Verificar alinhamento de índices
                    if len(current_resids) != len(lag_resids):
                        # Garantir alinhamento
                        common_index = lag_resids.index.intersection(current_resids.index)
                        if len(common_index) < 2:
                            st.warning("⚠️ Dados insuficientes para cálculo de half-life após alinhamento")
                            half_life = np.nan
                        else:
                            lag_resids = lag_resids[common_index]
                            current_resids = current_resids[common_index]
                    
                    # Calcular delta e prosseguir com cálculo de half-life
                    if len(lag_resids) >= 2 and len(current_resids) >= 2:
                        delta_resids = current_resids - lag_resids
                        
                        # Verificar se temos dados válidos
                        if not delta_resids.isna().all() and not lag_resids.isna().all():
                            # Remover NaNs que possam ter surgido na subtração
                            mask = ~np.isnan(delta_resids) & ~np.isnan(lag_resids)
                            delta_resids_clean = delta_resids[mask]
                            lag_resids_clean = lag_resids[mask]
                            
                            if len(delta_resids_clean) >= 2:
                                # Regressão para half-life com validações
                                x_hl = sm.add_constant(lag_resids_clean)
                                try:
                                    model_hl = sm.OLS(delta_resids_clean, x_hl).fit()
                                    
                                    # Validar coeficiente antes de calcular half-life
                                    if model_hl.params[1] < 0 and np.isfinite(model_hl.params[1]):
                                        half_life = -np.log(2) / model_hl.params[1]
                                        
                                        # Verificar se half-life é razoável
                                        if not np.isfinite(half_life) or half_life <= 0 or half_life > 365:
                                            st.info(f"ℹ️ Half-life calculado ({half_life:.1f} dias) fora de intervalo razoável")
                                            if half_life > 365:
                                                st.info("ℹ️ Half-life muito longo indica possível não-reversão à média")
                                    else:
                                        st.info("ℹ️ Coeficiente de reversão à média positivo - par não apresenta convergência")
                                        half_life = np.nan
                                except Exception as e:
                                    st.warning(f"⚠️ Erro no modelo OLS para half-life: {str(e)}")
                                    half_life = np.nan
                            else:
                                st.warning("⚠️ Dados insuficientes para cálculo de half-life após remoção de NaNs")
                                half_life = np.nan
                        else:
                            st.warning("⚠️ Todos os valores são NaN após cálculo de delta")
                            half_life = np.nan
                    else:
                        st.warning("⚠️ Dados insuficientes para cálculo de delta resíduos")
                        half_life = np.nan
                else:
                    st.warning("⚠️ Dados insuficientes após cálculo de lag")
                    half_life = np.nan
            else:
                st.warning("⚠️ Resíduos insuficientes para cálculo de half-life")
                half_life = np.nan
        except Exception as e:
            st.warning(f"⚠️ Erro no cálculo do half-life: {str(e)}")
            half_life = np.nan
            
        # Teste ADF para estacionariedade com validações
        try:
            if hasattr(model, 'resid') and len(model.resid) >= 7:  # ADF precisa de um mínimo de observações
                resids_for_adf = model.resid.copy()
                
                # Verificar valores não finitos
                if np.any(~np.isfinite(resids_for_adf)):
                    st.warning("⚠️ Resíduos contêm valores não finitos para teste ADF")
                    # Substituir valores não finitos por NaN
                    resids_for_adf[~np.isfinite(resids_for_adf)] = np.nan
                
                # Remover NaNs
                resids_for_adf_clean = pd.Series(resids_for_adf).dropna()
                
                if len(resids_for_adf_clean) >= 7:
                    adf_result = sm.tsa.stattools.adfuller(resids_for_adf_clean)
                    adf_pvalue = adf_result[1]
                    
                    # Validar resultado ADF
                    if not np.isfinite(adf_pvalue):
                        st.warning("⚠️ Resultado do teste ADF é um valor não finito")
                        adf_pvalue = np.nan
                else:
                    st.warning("⚠️ Dados insuficientes para teste ADF após remoção de NaNs")
                    adf_pvalue = np.nan
            else:
                st.warning("⚠️ Resíduos insuficientes para teste ADF")
                adf_pvalue = np.nan
        except Exception as e:
            st.warning(f"⚠️ Erro no teste ADF: {str(e)}")
            adf_pvalue = np.nan
    except Exception as e:        st.error(f"❌ Erro nos testes estatísticos: {str(e)}")
        # As variáveis já foram inicializadas com np.nan no início da função
    
    col1, col2, col3 = st.columns(3)
    
    # Exibir métricas com tratamento para NaN e validações adicionais
    with col1:
        if not np.isnan(coint_pvalue):
            color = "normal" if coint_pvalue >= 0.05 else "good"
            st.metric(
                "Teste de Cointegração (p-value)",
                f"{coint_pvalue:.4f}",
                "Cointegrado" if coint_pvalue < 0.05 else "Não cointegrado",
                delta_color=color
            )
        else:
            st.metric(
                "Teste de Cointegração (p-value)",
                "N/A"
            )
            st.info("ℹ️ Não foi possível realizar teste de cointegração")
    
    with col2:
        if not np.isnan(half_life):
            if half_life > 0 and half_life < 365:  # Valor razoável para half-life
                color = "good" if 2 <= half_life <= 30 else "normal"  # Half-life ideal entre 2-30 dias
                st.metric(
                    "Half-Life",
                    f"{half_life:.1f} dias",
                    delta_color=color
                )
                if half_life < 2:
                    st.info("ℹ️ Half-life muito curto - convergência muito rápida")
                elif half_life > 30:
                    st.info("ℹ️ Half-life longo - convergência lenta")
            else:
                st.metric(
                    "Half-Life",
                    f"{half_life:.1f} dias" if half_life > 0 else "Inválido"
                )
                st.warning("⚠️ Half-life fora de intervalo razoável")
        else:
            st.metric(
                "Half-Life",
                "N/A"
            )
            st.info("ℹ️ Não foi possível calcular half-life")
    
    with col3:
        if not np.isnan(adf_pvalue):
            color = "normal" if adf_pvalue >= 0.05 else "good"
            st.metric(
                "ADF Test (p-value)",
                f"{adf_pvalue:.4f}",
                "Estacionário" if adf_pvalue < 0.05 else "Não estacionário",
                delta_color=color
            )
        else:
            st.metric(
                "ADF Test (p-value)",
                "N/A"
            )
            st.info("ℹ️ Não foi possível realizar teste ADF")
    # Recomendações
    st.markdown("### 💡 Recomendações")
    
    # Criar um indicador de qualidade geral dos testes
    tests_quality = 0
    tests_count = 0
    reasons = []
    
    # Verificar cointegração
    if not np.isnan(coint_pvalue):
        tests_count += 1
        if coint_pvalue < 0.05:
            tests_quality += 1
            reasons.append("✅ Cointegração confirmada")
        else:
            reasons.append("⚠️ Cointegração não confirmada")
    else:
        reasons.append("⚠️ Teste de cointegração inconclusivo")
    
    # Verificar half-life
    if not np.isnan(half_life):
        tests_count += 1
        if 2 <= half_life <= 30:
            tests_quality += 1
            reasons.append(f"✅ Half-life adequado ({half_life:.1f} dias)")
        elif half_life > 0 and half_life < 100:
            tests_quality += 0.5
            if half_life < 2:
                reasons.append(f"ℹ️ Half-life muito curto ({half_life:.1f} dias)")
            else:
                reasons.append(f"ℹ️ Half-life longo ({half_life:.1f} dias)")
        else:
            reasons.append("⚠️ Half-life fora de intervalo aceitável")
    else:
        reasons.append("⚠️ Cálculo de half-life inconclusivo")
    
    # Verificar ADF
    if not np.isnan(adf_pvalue):
        tests_count += 1
        if adf_pvalue < 0.05:
            tests_quality += 1
            reasons.append("✅ Resíduos estacionários (ADF)")
        else:
            reasons.append("⚠️ Resíduos não estacionários (ADF)")
    else:
        reasons.append("⚠️ Teste ADF inconclusivo")
    
    # Verificar correlação
    try:
        if not np.isnan(corr):
            tests_count += 1
            if abs(corr) > 0.6:
                tests_quality += 1
                reasons.append(f"✅ Boa correlação ({corr:.3f})")
            elif abs(corr) > 0.3:
                tests_quality += 0.5
                reasons.append(f"ℹ️ Correlação moderada ({corr:.3f})")
            else:
                reasons.append(f"⚠️ Baixa correlação ({corr:.3f})")
    except NameError:
        # corr pode não estar definido se houve erro no cálculo
        pass
    
    # Calcular qualidade geral dos testes
    quality_score = tests_quality / max(tests_count, 1) if tests_count > 0 else 0
    
    # Recomendações baseadas na qualidade dos testes e resultados específicos
    if tests_count < 3:
        st.warning("""
        ⚠️ **Análise inconclusiva**
        
        Não foi possível realizar todos os testes estatísticos necessários devido a:
        - Dados insuficientes 
        - Valores ausentes ou inválidos
        - Problemas nos cálculos estatísticos
        
        **Sugestões:**
        - Utilize um período maior de dados históricos
        - Verifique a qualidade dos dados (valores ausentes, zeros, outliers)
        - Experimente outros pares de ativos com maior liquidez
        - Considere aplicar filtros ou transformações nos dados
        """)
    elif quality_score >= 0.75:
        # Lista os motivos da recomendação
        reasons_text = "\n".join([f"- {reason}" for reason in reasons if reason.startswith("✅")])
        
        st.success(f"""
        ✅ **Par adequado para trading estatístico**
        
        {reasons_text}
        
        **Próximos passos:**
        - Implemente uma estratégia de pair trading com este par
        - Defina os thresholds de entrada e saída baseados na análise estatística
        - Estabeleça um stop-loss adequado considerando a volatilidade do par
        - Monitore a estabilidade da relação estatística ao longo do tempo
        """)
    elif quality_score >= 0.5:
        # Lista os pontos positivos e questões a verificar
        positive_points = [reason for reason in reasons if reason.startswith("✅")]
        check_points = [reason for reason in reasons if reason.startswith("ℹ️") or reason.startswith("⚠️")]
        
        positive_text = "\n".join([f"- {point}" for point in positive_points])
        check_text = "\n".join([f"- {point}" for point in check_points])
        
        st.info(f"""
        ℹ️ **Par potencialmente adequado - requer análise adicional**
        
        **Pontos positivos:**
        {positive_text}
        
        **Pontos a verificar:**
        {check_text}
        
        **Recomendações:**
        - Realize backtests para validar a relação estatística
        - Considere períodos de tempo distintos para verificar a estabilidade
        - Ajuste os parâmetros de operação levando em conta as limitações identificadas
        """)
    else:
        # Lista os problemas principais
        warnings_text = "\n".join([f"- {reason}" for reason in reasons if reason.startswith("⚠️")])
        
        st.warning(f"""
        ⚠️ **Par não recomendado para pair trading estatístico**
        
        **Problemas identificados:**
        {warnings_text}
        
        **Alternativas:**
        - Analise outros pares com maior probabilidade de cointegração
        - Experimente diferentes períodos de dados
        - Considere outros métodos de análise estatística
        - Verifique se transformações (log, diferenças) melhoram a relação
        """)


def documentation_tab():
    """
    Tab 5: Documentação
    
    Fornece documentação técnica sobre as análises estatísticas
    implementadas no módulo, incluindo referências acadêmicas
    e descrição dos modelos utilizados.
    """
    st.subheader("📚 Documentação Técnica")
    
    st.markdown("""
    ### 📖 Análise Estatística Avançada
    
    Esta seção implementa diversos modelos estatísticos para análise
    de séries temporais financeiras. Abaixo estão as descrições dos
    principais modelos e metodologias utilizados.
    """)
    
    with st.expander("🎯 Análise de Eventos Extremos"):
        st.write("""
        #### 📉 Eventos Extremos
        
        **Metodologia**:
        - Estudo da probabilidade de eventos raros (large deviations)
        - Análise empírica de quedas extremas
        - Comparação com distribuição t-Student
        
        **Aplicação**:
        - Quantificação de risco de cauda (tail risk)
        - Estimação de probabilidades de drawdowns extremos
        - Simulação de cenários de estresse
        
        **Fórmulas-chave**:
        - Probabilidade empírica: `p = num_eventos_extremos / total_observações`
        - Estatísticas de ordem: análise de quantis empíricos
        """)
    
    with st.expander("📈 Comparação de Distribuições"):
        st.write("""
        #### 📊 Comparação Estatística
        
        **Testes Implementados**:
        - Kolmogorov-Smirnov: testa se duas amostras vêm da mesma distribuição
        - Mann-Whitney U: teste não-paramétrico para diferenças de medianas
        - Levene: teste para igualdade de variâncias
        - Anderson-Darling: teste de normalidade
        
        **Aplicação**:
        - Identificação de ativos com comportamentos estatísticos diferentes
        - Análise de mudanças de regime em séries temporais
        - Validação de hipóteses sobre distribuições de retornos
        """)
    
    with st.expander("🔬 Modelos de Risco"):
        st.write("""
        #### 📉 Value at Risk (VaR)
        
        **Metodologias**:
        - VaR Paramétrico: assumindo distribuição Normal
        - VaR Não-paramétrico: baseado em quantis empíricos
        - VaR com t-Student: para melhor modelagem de caudas pesadas
        
        **CVaR (Expected Shortfall)**:
        - Média das perdas além do VaR
        - Métrica coerente de risco
        - Mais sensível a eventos extremos que o VaR
        
        **Aplicação**:
        - Gestão de risco de mercado
        - Alocação de capital baseada em risco
        - Stress testing de portfolios
        """)
    
    with st.expander("🔄 Pair Trading"):
        st.write("""
        #### 📊 Pair Trading Estatístico
        
        **Metodologia**:
        - Testes de cointegração (Engle-Granger)
        - Cálculo de half-life para média-reversão
        - Modelagem de spread estatístico
        
        **Métricas-chave**:
        - p-value do teste de cointegração
        - Half-life de reversão à média
        - Correlação entre retornos
        
        **Aplicação**:
        - Estratégias de arbitragem estatística
        - Trading market-neutral
        - Hedge de exposições direcionais
        """)
    
    st.markdown("### 📋 Referências Bibliográficas")
    
    st.write("""
    1. Alexander, C. (2008). *Market Risk Analysis*. Wiley.
    2. Tsay, R.S. (2010). *Analysis of Financial Time Series*. Wiley.
    3. McNeil, A.J., Frey, R., & Embrechts, P. (2015). *Quantitative Risk Management*. Princeton University Press.
    4. Vidyamurthy, G. (2004). *Pairs Trading: Quantitative Methods and Analysis*. Wiley.
    5. Rodrigues, C.A. (2023). *Market-Neutral Portfolios: A Solution Based on Automated Strategies*. GLOBAL JOURNAL OF RESEARCHES IN ENGINEERING, v. 23, p. 1-10.
    """)
    
    st.markdown("### ⚠️ Notas e Limitações")
    
    st.info("""
    **Limitações dos Modelos**:
    
    - Modelos estatísticos baseiam-se em dados históricos, que podem não representar condições futuras
    - Distribuições de retornos financeiros frequentemente exibem caudas mais pesadas que as distribuições teóricas
    - Correlações e outras relações estatísticas podem mudar drasticamente em períodos de estresse de mercado
    
    Recomenda-se complementar estas análises estatísticas com análise fundamentalista e conhecimento do contexto macroeconômico.
    """)
    
def _display_asset_extreme_analysis(stat_analyzer, df, asset_symbol=PETR4_SYMBOL, threshold=0.10):
    """
    Exibe análise de eventos extremos para um ativo específico.
    Esta função tenta usar asset_extreme_analysis se existir, ou petrobras_extreme_analysis como fallback.
    Se nenhum método estiver disponível, implementa uma análise básica.
    
    Args:
        stat_analyzer: Instância de StatisticalAnalysis 
        df: DataFrame de preços
        asset_symbol: Símbolo do ativo a analisar (padrão: PETR4_SYMBOL)
        threshold: Threshold de queda para considerar extremo (padrão: 10%)
    """
    try:
        # Verificar se existe o método asset_extreme_analysis, caso contrário usar petrobras_extreme_analysis
        if hasattr(stat_analyzer, 'asset_extreme_analysis'):
            extreme_analysis = stat_analyzer.asset_extreme_analysis(asset_symbol=asset_symbol, threshold=threshold)
        elif asset_symbol == PETR4_SYMBOL and hasattr(stat_analyzer, 'petrobras_extreme_analysis'):
            extreme_analysis = stat_analyzer.petrobras_extreme_analysis(threshold=threshold)
        else:
            # Criar análise simplificada
            try:
                asset_returns = df[asset_symbol].pct_change().dropna()
                if len(asset_returns) == 0:
                    raise ValueError("Dados insuficientes para análise")
                    
                extreme_falls = asset_returns[asset_returns <= -threshold]
                extreme_analysis = {
                    'asset_symbol': asset_symbol,
                    'total_days': len(asset_returns),
                    'extreme_falls_count': len(extreme_falls),
                    'probability': len(extreme_falls) / len(asset_returns) if len(asset_returns) > 0 else 0,
                    'daily_statistics': {
                        'mean': asset_returns.mean(),
                        'std': asset_returns.std(),
                        'skewness': skew(asset_returns) if len(asset_returns) > 3 else 0,
                        'kurtosis': kurtosis(asset_returns, fisher=True) if len(asset_returns) > 4 else 0,
                    }
                }
            except Exception as e:
                st.error(f"Erro ao calcular estatísticas: {str(e)}")
                st.info("Implementando análise mínima de fallback.")
                
                # Versão de emergência com dados mínimos
                extreme_analysis = {
                    'asset_symbol': asset_symbol,
                    'total_days': df.shape[0] if df is not None else 0,
                    'extreme_falls_count': 0,
                    'probability': 0,
                    'daily_statistics': {
                        'mean': 0,
                        'std': 0,
                        'skewness': 0,
                        'kurtosis': 0,
                    }
                }
            
        # Exibir resultados
        st.subheader(f"📉 Análise de Eventos Extremos - {asset_symbol}")
        
        # Obter estatísticas
        daily_stats = extreme_analysis.get('daily_statistics', {})
        extreme_stats = extreme_analysis.get('extreme_statistics', {})
        recovery_stats = extreme_analysis.get('recovery_statistics', {})
        prob_empirical = extreme_analysis.get('probability', 0)
        total_days = extreme_analysis.get('total_days', 0)
        extreme_count = extreme_analysis.get('extreme_falls_count', 0)
        extreme_dates = extreme_stats.get('dates', [])
        
        # Métricas principais
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Prob. Empírica", f"{prob_empirical:.2%}")
        with col2:
            st.metric("Eventos Extremos", f"{extreme_count}/{total_days}")
        with col3:
            daily_vol = daily_stats.get('std', 0)
            annual_vol = daily_vol * np.sqrt(252)
            st.metric("Volatilidade Anual", f"{annual_vol:.1%}")
        with col4:
            skewness = daily_stats.get('skewness', 0)
            kurtosis_value = daily_stats.get('kurtosis', 0)
            st.metric("Assimetria/Curtose", f"{skewness:.2f}/{kurtosis_value:.2f}")
        
        # Análise detalhada de distribuição de probabilidade
        st.markdown("#### 📊 Distribuição de Retornos e Probabilidade de Eventos Extremos")
        
        # Criar tabs para diferentes análises
        tab_empirical, tab_normal, tab_tstudent, tab_historico = st.tabs([
            "📊 Empírica", 
            "🔄 Normal", 
            "📈 t-Student", 
            "🗓️ Histórico"
        ])
        
        # Preparar dados e distribuições teóricas
        try:
            asset_returns = df[asset_symbol].pct_change().dropna()
            
            with tab_empirical:
                st.markdown("##### Análise Empírica")
                
                # Criar histograma com densidade
                fig = go.Figure()
                
                # Calcular bins de forma dinâmica
                bin_width = (asset_returns.max() - asset_returns.min()) / 40  # 40 bins é um bom compromisso
                
                # Adicionar histograma com densidade de probabilidade
                fig.add_trace(go.Histogram(
                    x=asset_returns,
                    histnorm='probability density',
                    name="Retornos",
                    opacity=0.6,
                    marker_color='#1f77b4'
                ))
                
                # Adicionar linha vertical para o threshold
                fig.add_vline(
                    x=-threshold, 
                    line_dash="dash", 
                    line_color="red",
                    annotation_text=f"Threshold: -{threshold:.0%}",
                    annotation_position="top right"
                )
                
                # Customizar layout
                fig.update_layout(
                    title=f"Distribuição Empírica de Retornos: {asset_symbol}",
                    xaxis_title="Retorno Diário",
                    yaxis_title="Densidade de Probabilidade",
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Mostrar probabilidade empirica com mais detalhes
                if extreme_count > 0:
                    st.info(f"""
                    📊 **Probabilidade Empírica**: {prob_empirical:.2%}
                    
                    Baseado em dados históricos, a probabilidade de {asset_symbol} ter uma queda diária superior a {threshold:.0%} é de {prob_empirical:.2%}.
                    Isso equivale a aproximadamente 1 queda a cada {1/prob_empirical:.0f} dias de negociação, ou cerca de {252/prob_empirical:.1f} dias úteis por ano.
                    """)
                else:
                    st.info("Não foram observadas quedas superiores ao threshold no período analisado.")
            
            with tab_normal:
                st.markdown("##### Modelagem com Distribuição Normal")
                
                # Parâmetros da distribuição Normal
                mu = asset_returns.mean()
                sigma = asset_returns.std()
                
                # Probabilidade teórica baseada na Normal
                prob_normal = stats.norm.cdf(-threshold, mu, sigma)
                
                # Criar gráfico
                x = np.linspace(asset_returns.min(), asset_returns.max(), 1000)
                y = stats.norm.pdf(x, mu, sigma)
                
                fig = go.Figure()
                
                # Adicionar histograma com densidade de probabilidade
                fig.add_trace(go.Histogram(
                    x=asset_returns,
                    histnorm='probability density',
                    name="Retornos",
                    opacity=0.6,
                    marker_color='#1f77b4'
                ))
                
                # Adicionar curva de distribuição normal
                fig.add_trace(go.Scatter(
                    x=x,
                    y=y,
                    mode='lines',
                    name='Normal',
                    line=dict(color='red', width=2)
                ))
                
                # Área sombreada para quedas extremas
                x_extreme = np.linspace(asset_returns.min(), -threshold, 100)
                y_extreme = stats.norm.pdf(x_extreme, mu, sigma)
                
                fig.add_trace(go.Scatter(
                    x=x_extreme,
                    y=y_extreme,
                    fill='tozeroy',
                    fillcolor='rgba(255,0,0,0.2)',
                    line=dict(color='rgba(255,0,0,0)'),
                    name=f'Prob. Normal: {prob_normal:.2%}'
                ))
                
                # Adicionar linha vertical para o threshold
                fig.add_vline(
                    x=-threshold, 
                    line_dash="dash", 
                    line_color="red",
                    annotation_text=f"Threshold: -{threshold:.0%}",
                    annotation_position="top right"
                )
                
                # Customizar layout
                fig.update_layout(
                    title=f"Modelagem com Distribuição Normal: {asset_symbol}",
                    xaxis_title="Retorno Diário",
                    yaxis_title="Densidade de Probabilidade",
                    showlegend=True
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Mostrar comparação entre probabilidade empírica e teórica
                ratio = prob_empirical / prob_normal if prob_normal > 0 else 0
                
                if ratio > 1.3:
                    st.warning(f"""
                    ⚠️ **Alerta**: A probabilidade empírica ({prob_empirical:.2%}) é {ratio:.1f}x maior que a estimada pela distribuição Normal ({prob_normal:.2%}).
                    
                    Isso indica que o ativo possui **caudas mais pesadas** do que o previsto pela Normal, subestimando o risco de eventos extremos.
                    """)
                elif ratio < 0.7 and ratio > 0:
                    st.info(f"""
                    ℹ️ **Observação**: A probabilidade empírica ({prob_empirical:.2%}) é {1/ratio:.1f}x menor que a estimada pela distribuição Normal ({prob_normal:.2%}).
                    
                    Isso pode indicar que o período analisado teve menos eventos extremos do que o esperado teoricamente.
                    """)
                else:
                    st.success(f"""
                    ✅ **Validação**: A probabilidade empírica ({prob_empirical:.2%}) é relativamente próxima da estimada pela distribuição Normal ({prob_normal:.2%}).
                    
                    A modelagem Normal captura razoavelmente bem o comportamento de quedas do ativo neste threshold.
                    """)
            
            with tab_tstudent:
                st.markdown("##### Modelagem com Distribuição t-Student")
                
                # Estimar parâmetros da t-Student (graus de liberdade)
                def t_loglikelihood(params, data):
                    df, loc, scale = params
                    return -np.sum(stats.t.logpdf(data, df=df, loc=loc, scale=scale))
                
                # Estimativa inicial baseada em momentos
                initial_params = [6, asset_returns.mean(), asset_returns.std()]
                
                try:
                    # Usar otimização para encontrar melhores parâmetros
                    from scipy.optimize import minimize
                    result = minimize(t_loglikelihood, initial_params, args=(asset_returns,), 
                                     bounds=[(2.1, 50), (None, None), (0.0001, None)])
                    
                    df_param, loc_param, scale_param = result.x
                    
                    # Probabilidade teórica baseada na t-Student
                    prob_t = stats.t.cdf(-threshold, df=df_param, loc=loc_param, scale=scale_param)
                    
                    # Criar gráfico
                    x = np.linspace(asset_returns.min(), asset_returns.max(), 1000)
                    y_t = stats.t.pdf(x, df=df_param, loc=loc_param, scale=scale_param)
                    y_norm = stats.norm.pdf(x, mu, sigma)
                    
                    fig = go.Figure()
                    
                    # Adicionar histograma com densidade de probabilidade
                    fig.add_trace(go.Histogram(
                        x=asset_returns,
                        histnorm='probability density',
                        name="Retornos",
                        opacity=0.4,
                        marker_color='#1f77b4'
                    ))
                    
                    # Adicionar curva de distribuição t-Student
                    fig.add_trace(go.Scatter(
                        x=x,
                        y=y_t,
                        mode='lines',
                        name='t-Student',
                        line=dict(color='red', width=2)
                    ))
                    
                    # Adicionar curva de distribuição normal para comparação
                    fig.add_trace(go.Scatter(
                        x=x,
                        y=y_norm,
                        mode='lines',
                        name='Normal',
                        line=dict(color='green', width=2, dash='dash')
                    ))
                    
                    # Área sombreada para quedas extremas (t-Student)
                    x_extreme = np.linspace(asset_returns.min(), -threshold, 100)
                    y_extreme = stats.t.pdf(x_extreme, df=df_param, loc=loc_param, scale=scale_param)
                    
                    fig.add_trace(go.Scatter(
                        x=x_extreme,
                        y=y_extreme,
                        fill='tozeroy',
                        fillcolor='rgba(255,0,0,0.2)',
                        line=dict(color='rgba(255,0,0,0)'),
                        name=f'Prob. t-Student: {prob_t:.2%}'
                    ))
                    
                    # Adicionar linha vertical para o threshold
                    fig.add_vline(
                        x=-threshold, 
                        line_dash="dash", 
                        line_color="red",
                        annotation_text=f"Threshold: -{threshold:.0%}",
                        annotation_position="top right"
                    )
                    
                    # Customizar layout
                    fig.update_layout(
                        title=f"Modelagem com Distribuição t-Student (v={df_param:.1f}): {asset_symbol}",
                        xaxis_title="Retorno Diário",
                        yaxis_title="Densidade de Probabilidade",
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Mostrar comparação entre probabilidades
                    ratio_t = prob_empirical / prob_t if prob_t > 0 else 0                    # Tabela comparativa - com tratamento para evitar nan%
                    prob_normal_display = "0.00%" if np.isnan(prob_normal) else f"{prob_normal:.2%}"
                    prob_t_display = "0.00%" if np.isnan(prob_t) else f"{prob_t:.2%}" 
                    
                    normal_ratio = "0.00" if np.isnan(prob_normal) or prob_empirical == 0 else f"{prob_normal/prob_empirical:.2f}"
                    t_ratio = "0.00" if np.isnan(prob_t) or prob_empirical == 0 else f"{prob_t/prob_empirical:.2f}"
                    
                    comp_df = pd.DataFrame({
                        "Modelo": ["Empírico", "Normal", "t-Student"],
                        "Probabilidade": [f"{prob_empirical:.2%}", prob_normal_display, prob_t_display],
                        "Razão p/ Empírico": ["1.00", normal_ratio, t_ratio]
                    })
                    
                    st.table(comp_df)
                    
                    # Adicionar botão de download para esta análise específica
                    try:
                        from pdf_export_helpers import add_download_buttons_to_extreme_analysis
                        add_download_buttons_to_extreme_analysis(
                            asset_symbol, threshold, prob_empirical, prob_normal, prob_t, df_param
                        )
                    except Exception as e:
                        st.warning(f"Não foi possível adicionar opção de download: {str(e)}")
                    
                    if abs(ratio_t - 1) < 0.2 and ratio_t > 0:
                        st.success(f"""
                        ✅ **Validação**: A distribuição t-Student com {df_param:.1f} graus de liberdade modela bem os eventos extremos deste ativo.
                        
                        A probabilidade estimada pela t-Student ({prob_t:.2%}) está muito próxima da probabilidade empírica ({prob_empirical:.2%}).
                        """)
                    elif ratio_t > 1:
                        st.warning(f"""
                        ⚠️ **Alerta**: A probabilidade empírica ({prob_empirical:.2%}) ainda é maior que a estimada pela t-Student ({prob_t:.2%}).
                        
                        Isso sugere que mesmo a modelagem com t-Student pode estar subestimando o risco de quedas extremas neste ativo.
                        """)
                    else:
                        st.info(f"""
                        ℹ️ **Observação**: A modelagem com t-Student ({prob_t:.2%}) fornece uma estimativa mais conservadora que a probabilidade empírica ({prob_empirical:.2%}).
                        
                        Isso pode ser adequado para modelagem de risco com margem de segurança.
                        """)
                        
                except Exception as e:
                    st.warning(f"Não foi possível estimar os parâmetros da distribuição t-Student: {str(e)}")
                    st.info("Verifique se há dados suficientes ou tente novamente com um conjunto de dados maior.")
            
            with tab_historico:
                st.markdown("##### Datas de Quedas Extremas")
                
                if extreme_dates and len(extreme_dates) > 0:
                    # Converter timestamps para strings formatadas
                    if isinstance(extreme_dates[0], pd.Timestamp):
                        date_strings = [date.strftime('%d/%m/%Y') for date in extreme_dates]
                    else:
                        date_strings = extreme_dates
                    
                    # Obter os retornos para essas datas
                    extreme_returns_values = []
                    for date in extreme_dates:
                        try:
                            if date in asset_returns.index:
                                extreme_returns_values.append(asset_returns[date])
                        except:
                            extreme_returns_values.append(None)
                    
                    # Criar dataframe para exibição
                    extreme_df = pd.DataFrame({
                        "Data": date_strings,
                        "Queda (%)": [f"{ret*100:.2f}%" if ret is not None else "N/A" for ret in extreme_returns_values]
                    })
                    
                    st.dataframe(extreme_df, use_container_width=True)
                    
                    # Recuperação média
                    mean_recovery = recovery_stats.get('mean_days', None)
                    if mean_recovery:
                        st.metric("Tempo Médio de Recuperação", f"{mean_recovery:.1f} dias")
                        
                        recovery_rate = recovery_stats.get('recovery_rate', None)
                        if recovery_rate:
                            st.info(f"""
                            📈 **Recuperação**: {recovery_rate:.0%} das quedas superiores a {threshold:.0%} se recuperaram em até 30 dias.
                            
                            O tempo médio de recuperação é de {mean_recovery:.1f} dias após uma queda extrema.
                            """)
                else:
                    st.info("Não foram observadas quedas superiores ao threshold no período analisado.")
                    
                # Oferecer botão para baixar dados de quedas extremas
                if extreme_dates and len(extreme_dates) > 0:
                    csv = extreme_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Baixar Dados de Quedas Extremas",
                        data=csv,
                        file_name=f'quedas_extremas_{asset_symbol}.csv',
                        mime='text/csv',
                    )
        
        except Exception as e:
            st.error(f"Erro ao gerar análise detalhada: {str(e)}")
        
        # Exibir interpretação
        st.markdown("#### 💡 Interpretação dos Resultados")
        
        # Determinar o modelo mais adequado para esta análise
        best_model = "empírico"
        best_prob = prob_empirical
        
        if 'prob_t' in locals() and abs(ratio_t - 1) < 0.2:
            best_model = "t-Student"
            best_prob = prob_t
        elif 'prob_normal' in locals() and abs(ratio - 1) < 0.2:
            best_model = "Normal"
            best_prob = prob_normal
            
        # Interpretar em termos práticos
        if best_prob > 0.05:  # 5%
            st.warning(f"""
            ⚠️ **Alto Risco**: Baseado no modelo {best_model}, a probabilidade de {best_prob:.2%} para quedas diárias superiores a {threshold:.0%} 
            indica volatilidade elevada.
            
            **Recomendações:**
            - Considere estratégias de hedge (opções de venda, stop-loss)
            - Diversifique o portfólio para reduzir exposição
            - Monitore atentamente fatores externos que podem amplificar quedas
            """)
        elif best_prob > 0.02:  # 2%
            st.info(f"""
            💡 **Risco Moderado**: Baseado no modelo {best_model}, a probabilidade de {best_prob:.2%} para quedas diárias superiores a {threshold:.0%}
            é significativa.
            
            **Recomendações:**
            - Monitore indicadores macro e setoriais que podem afetar o ativo
            - Mantenha um plano de contingência para eventos negativos
            - Considere um mix de posições de longo prazo e proteções táticas
            """)
        else:
            st.success(f"""
            ✅ **Risco Controlado**: Baseado no modelo {best_model}, a probabilidade de {best_prob:.2%} para quedas diárias superiores a {threshold:.0%}
            é relativamente baixa no horizonte analisado.
            
            **Recomendações:**
            - Mantenha monitoramento regular dos indicadores de risco
            - Reavalie periodicamente essa análise, especialmente após mudanças de mercado significativas
            - Considere este ativo para estratégias de longo prazo com uma tolerância controlada ao risco
            """)
        
    except Exception as e:
        st.error(f"Erro na análise de eventos extremos: {str(e)}")
        st.info("Falha ao executar análise detalhada. Verifique os dados ou tente com um período mais longo.")
