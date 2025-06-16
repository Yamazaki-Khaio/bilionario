"""
Módulo de Análise de Normalidade Avançada
Implementa testes de normalidade, normalização T-Student e Q-Q plots
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import (shapiro, jarque_bera, kstest, anderson, normaltest,
                        t, norm, probplot, chi2_contingency)
import matplotlib.pyplot as plt
from financial_formatting import format_percentage, format_ratio
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")


class NormalityAnalysis:
    """Classe para análise avançada de normalidade de dados financeiros"""
    
    def __init__(self, data):
        """
        Inicializa a análise de normalidade
        
        Args:
            data (pd.DataFrame or pd.Series): Dados para análise
        """
        if isinstance(data, pd.DataFrame):
            # Filtrar apenas colunas numéricas e converter para numeric se necessário
            self.data = data.copy()
            
            # Converter colunas para numérico, transformando não-numéricos em NaN
            for col in self.data.columns:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
            
            # Filtrar apenas colunas que têm pelo menos alguns valores numéricos
            numeric_columns = []
            for col in self.data.columns:
                if self.data[col].notna().sum() > 10:  # Pelo menos 10 valores válidos
                    numeric_columns.append(col)
            
            if numeric_columns:
                self.data = self.data[numeric_columns]
                
                # Calcular retornos apenas se há dados suficientes
                if len(self.data) > 1:
                    self.returns = self.data.pct_change().dropna()
                else:
                    self.returns = pd.DataFrame()
            else:
                self.data = pd.DataFrame()
                self.returns = pd.DataFrame()
        else:
            # Para Series, também garantir que é numérico
            self.data = pd.to_numeric(data, errors='coerce').dropna()
            self.returns = self.data
            
    def t_student_normalization(self, series, method='standardized'):
        """
        Normalização usando distribuição T-Student
        
        Args:
            series (pd.Series): Série temporal
            method (str): Método de normalização ('standardized', 'probability', 'percentile')
            
        Returns:
            dict: Dados normalizados e parâmetros
        """
        clean_data = series.dropna()
        
        if len(clean_data) < 10:
            return {"error": "Dados insuficientes para normalização T-Student"}
        
        try:
            # Ajustar distribuição T-Student
            t_params = stats.t.fit(clean_data)
            df_param, loc_param, scale_param = t_params
            
            if method == 'standardized':
                # Normalização padronizada
                normalized = (clean_data - loc_param) / scale_param
                
            elif method == 'probability':
                # Transformação para probabilidades
                normalized = stats.t.cdf(clean_data, df_param, loc_param, scale_param)
                
            elif method == 'percentile':
                # Transformação para percentis
                normalized = stats.t.cdf(clean_data, df_param, loc_param, scale_param) * 100
                
            # Teste de normalidade dos dados normalizados
            if len(normalized) >= 8:
                shapiro_stat, shapiro_p = shapiro(normalized)
                jb_stat, jb_p = jarque_bera(normalized)
            else:
                shapiro_stat = shapiro_p = jb_stat = jb_p = np.nan
            
            return {
                'normalized_data': normalized,
                'original_data': clean_data,
                't_parameters': {
                    'degrees_freedom': df_param,
                    'location': loc_param,
                    'scale': scale_param
                },
                'method': method,
                'normality_tests_normalized': {
                    'shapiro': {'statistic': shapiro_stat, 'p_value': shapiro_p},
                    'jarque_bera': {'statistic': jb_stat, 'p_value': jb_p}
                },
                'statistics_normalized': {
                    'mean': normalized.mean(),
                    'std': normalized.std(),
                    'skewness': stats.skew(normalized),
                    'kurtosis': stats.kurtosis(normalized, fisher=True)
                }
            }
            
        except Exception as e:
            return {"error": f"Erro na normalização T-Student: {str(e)}"}
    
    def comprehensive_normality_tests(self, series, alpha=0.05):
        """
        Executa bateria completa de testes de normalidade
        
        Args:
            series (pd.Series): Série temporal
            alpha (float): Nível de significância
            
        Returns:
            dict: Resultados de todos os testes
        """
        clean_data = series.dropna()
        
        if len(clean_data) < 8:
            return {"error": "Dados insuficientes para testes de normalidade"}
        
        results = {
            'sample_size': len(clean_data),
            'alpha': alpha,
            'tests': {}
        }
        
        # 1. Teste de Shapiro-Wilk
        try:
            if len(clean_data) <= 5000:  # Limitação do teste
                shapiro_stat, shapiro_p = shapiro(clean_data)
                results['tests']['shapiro_wilk'] = {
                    'statistic': shapiro_stat,
                    'p_value': shapiro_p,
                    'is_normal': shapiro_p > alpha,
                    'description': 'Teste mais poderoso para amostras pequenas (<5000)'
                }
        except Exception:
            results['tests']['shapiro_wilk'] = {'error': 'Falha no teste'}
        
        # 2. Teste de Jarque-Bera
        try:
            jb_stat, jb_p = jarque_bera(clean_data)
            results['tests']['jarque_bera'] = {
                'statistic': jb_stat,
                'p_value': jb_p,
                'is_normal': jb_p > alpha,
                'description': 'Baseado em assimetria e curtose'
            }
        except Exception:
            results['tests']['jarque_bera'] = {'error': 'Falha no teste'}
        
        # 3. Teste de Kolmogorov-Smirnov
        try:
            mean, std = clean_data.mean(), clean_data.std()
            ks_stat, ks_p = kstest(clean_data, lambda x: norm.cdf(x, mean, std))
            results['tests']['kolmogorov_smirnov'] = {
                'statistic': ks_stat,
                'p_value': ks_p,
                'is_normal': ks_p > alpha,
                'description': 'Compara distribuição empírica com normal'
            }
        except Exception:
            results['tests']['kolmogorov_smirnov'] = {'error': 'Falha no teste'}
        
        # 4. Teste de Anderson-Darling
        try:
            ad_result = anderson(clean_data, dist='norm')
            # Usar nível de significância mais próximo
            critical_values = ad_result.critical_values
            significance_levels = ad_result.significance_level
            
            # Encontrar valor crítico para alpha
            if alpha == 0.05:
                idx = 2  # 5%
            elif alpha == 0.01:
                idx = 4  # 1%
            else:
                idx = 2  # Default 5%
            
            if idx < len(critical_values):
                is_normal = ad_result.statistic < critical_values[idx]
            else:
                is_normal = None
                
            results['tests']['anderson_darling'] = {
                'statistic': ad_result.statistic,
                'critical_values': critical_values.tolist(),
                'significance_levels': significance_levels.tolist(),
                'is_normal': is_normal,
                'description': 'Versão melhorada do KS, mais sensível nas caudas'
            }
        except Exception:
            results['tests']['anderson_darling'] = {'error': 'Falha no teste'}
        
        # 5. Teste de D'Agostino-Pearson
        try:
            if len(clean_data) >= 20:
                dp_stat, dp_p = normaltest(clean_data)
                results['tests']['dagostino_pearson'] = {
                    'statistic': dp_stat,
                    'p_value': dp_p,
                    'is_normal': dp_p > alpha,
                    'description': 'Combina testes de assimetria e curtose'
                }
        except Exception:
            results['tests']['dagostino_pearson'] = {'error': 'Falha no teste'}
        
        # 6. Teste de Lilliefors (se disponível)
        try:
            # Implementação manual simplificada do teste de Lilliefors
            n = len(clean_data)
            sorted_data = np.sort(clean_data)
            mean, std = clean_data.mean(), clean_data.std(ddof=1)
            
            # Função de distribuição empírica
            ecdf = np.arange(1, n + 1) / n
            
            # Função de distribuição normal estimada
            ncdf = norm.cdf(sorted_data, mean, std)
            
            # Estatística de Lilliefors
            D_plus = np.max(ecdf - ncdf)
            D_minus = np.max(ncdf - (np.arange(0, n) / n))
            lillie_stat = max(D_plus, D_minus)
            
            # Valor crítico aproximado (Lilliefors, 1967)
            if n >= 30:
                critical_value = 0.886 / np.sqrt(n)  # Para alpha = 0.05
            else:
                critical_value = None
            
            results['tests']['lilliefors'] = {
                'statistic': lillie_stat,
                'critical_value': critical_value,
                'is_normal': lillie_stat < critical_value if critical_value else None,
                'description': 'KS modificado para parâmetros estimados'
            }
        except Exception:
            results['tests']['lilliefors'] = {'error': 'Falha no teste'}
        
        # Resumo dos resultados
        normal_tests = [test for test in results['tests'].values() 
                       if 'is_normal' in test and test['is_normal'] is not None]
        
        if normal_tests:
            normal_count = sum(1 for test in normal_tests if test['is_normal'])
            total_tests = len(normal_tests)
            
            results['summary'] = {
                'tests_indicating_normal': normal_count,
                'total_valid_tests': total_tests,
                'proportion_normal': normal_count / total_tests,
                'consensus': normal_count / total_tests > 0.5,
                'recommendation': self._get_normality_recommendation(results['tests'])
            }
        
        return results
    
    def _get_normality_recommendation(self, tests):
        """
        Gera recomendação baseada nos resultados dos testes
        
        Args:
            tests (dict): Resultados dos testes
            
        Returns:
            str: Recomendação
        """
        # Priorizar testes mais robustos
        priority_tests = ['shapiro_wilk', 'anderson_darling', 'jarque_bera']
        
        normal_votes = 0
        total_votes = 0
        
        for test_name in priority_tests:
            if test_name in tests and 'is_normal' in tests[test_name]:
                if tests[test_name]['is_normal'] is not None:
                    total_votes += 1
                    if tests[test_name]['is_normal']:
                        normal_votes += 1
        
        if total_votes == 0:
            return "Dados insuficientes para recomendação"
        
        proportion = normal_votes / total_votes
        
        if proportion >= 0.7:
            return "Dados seguem distribuição normal"
        elif proportion >= 0.3:
            return "Evidência mista - considerar distribuições alternativas"
        else:
            return "Dados não seguem distribuição normal - usar distribuição T-Student ou não-paramétricos"
    
    def enhanced_qq_plots(self, series, distributions=['norm', 't', 'lognorm', 'gamma']):
        """
        Cria Q-Q plots para múltiplas distribuições
        
        Args:
            series (pd.Series): Série temporal
            distributions (list): Lista de distribuições para testar
            
        Returns:
            dict: Figuras dos Q-Q plots e estatísticas de ajuste
        """
        clean_data = series.dropna()
        
        if len(clean_data) < 10:
            return {"error": "Dados insuficientes para Q-Q plots"}
        
        results = {
            'plots': {},
            'fit_statistics': {},
            'best_fit': None,
            'data_size': len(clean_data)
        }
        
        best_r_squared = -1
        
        for dist_name in distributions:
            try:
                # Criar figura para cada distribuição
                fig = go.Figure()
                
                if dist_name == 'norm':
                    # Q-Q plot normal
                    theoretical_quantiles, sample_quantiles = probplot(clean_data, dist='norm', plot=None)
                    
                    # Calcular R²
                    r_squared = np.corrcoef(theoretical_quantiles, sample_quantiles)[0, 1] ** 2
                    
                elif dist_name == 't':
                    # Ajustar distribuição T e criar Q-Q plot
                    t_params = stats.t.fit(clean_data)
                    theoretical_quantiles, sample_quantiles = probplot(
                        clean_data, dist=stats.t, sparams=t_params, plot=None
                    )
                    r_squared = np.corrcoef(theoretical_quantiles, sample_quantiles)[0, 1] ** 2
                    
                elif dist_name == 'lognorm':
                    # Para lognormal, usar apenas valores positivos
                    positive_data = clean_data[clean_data > 0]
                    if len(positive_data) > 10:
                        lognorm_params = stats.lognorm.fit(positive_data, floc=0)
                        theoretical_quantiles, sample_quantiles = probplot(
                            positive_data, dist=stats.lognorm, sparams=lognorm_params, plot=None
                        )
                        r_squared = np.corrcoef(theoretical_quantiles, sample_quantiles)[0, 1] ** 2
                    else:
                        continue
                        
                elif dist_name == 'gamma':
                    # Para gamma, usar apenas valores positivos
                    positive_data = clean_data[clean_data > 0]
                    if len(positive_data) > 10:
                        gamma_params = stats.gamma.fit(positive_data, floc=0)
                        theoretical_quantiles, sample_quantiles = probplot(
                            positive_data, dist=stats.gamma, sparams=gamma_params, plot=None
                        )
                        r_squared = np.corrcoef(theoretical_quantiles, sample_quantiles)[0, 1] ** 2
                    else:
                        continue
                
                # Adicionar pontos do Q-Q plot
                fig.add_trace(go.Scatter(
                    x=theoretical_quantiles,
                    y=sample_quantiles,
                    mode='markers',
                    name=f'Q-Q Plot {dist_name.title()}',
                    marker=dict(size=4, opacity=0.6)
                ))
                
                # Adicionar linha de referência (y=x)
                min_val = min(min(theoretical_quantiles), min(sample_quantiles))
                max_val = max(max(theoretical_quantiles), max(sample_quantiles))
                
                fig.add_trace(go.Scatter(
                    x=[min_val, max_val],
                    y=[min_val, max_val],
                    mode='lines',
                    name='Linha de Referência',
                    line=dict(color='red', dash='dash')
                ))
                
                fig.update_layout(
                    title=f'Q-Q Plot - Distribuição {dist_name.title()} (R² = {r_squared:.4f})',
                    xaxis_title=f'Quantis Teóricos ({dist_name.title()})',
                    yaxis_title='Quantis da Amostra',
                    showlegend=True,
                    width=600,
                    height=500
                )
                
                results['plots'][dist_name] = fig
                results['fit_statistics'][dist_name] = {
                    'r_squared': r_squared,
                    'correlation': np.corrcoef(theoretical_quantiles, sample_quantiles)[0, 1]
                }
                
                # Atualizar melhor ajuste
                if r_squared > best_r_squared:
                    best_r_squared = r_squared
                    results['best_fit'] = {
                        'distribution': dist_name,
                        'r_squared': r_squared,
                        'correlation': np.corrcoef(theoretical_quantiles, sample_quantiles)[0, 1]
                    }
                
            except Exception as e:
                results['fit_statistics'][dist_name] = {'error': str(e)}
        
        return results
    
    def simulation_based_normality_test(self, series, n_simulations=1000, alpha=0.05):
        """
        Teste de normalidade baseado em simulação Monte Carlo
        
        Args:
            series (pd.Series): Série temporal
            n_simulations (int): Número de simulações
            alpha (float): Nível de significância
            
        Returns:
            dict: Resultados do teste por simulação
        """
        clean_data = series.dropna()
        
        if len(clean_data) < 20:
            return {"error": "Dados insuficientes para teste por simulação"}
        
        n = len(clean_data)
        mean_obs = clean_data.mean()
        std_obs = clean_data.std()
        
        # Calcular estatísticas da amostra original
        original_stats = {
            'jarque_bera': jarque_bera(clean_data)[0],
            'shapiro': shapiro(clean_data)[0] if n <= 5000 else None,
            'skewness': abs(stats.skew(clean_data)),
            'kurtosis': abs(stats.kurtosis(clean_data, fisher=True))
        }
        
        # Simulações
        simulation_stats = {stat: [] for stat in original_stats.keys() if original_stats[stat] is not None}
        
        for _ in range(n_simulations):
            # Gerar amostra normal com mesmos parâmetros
            sim_data = np.random.normal(mean_obs, std_obs, n)
            
            # Calcular estatísticas para amostra simulada
            if original_stats['jarque_bera'] is not None:
                simulation_stats['jarque_bera'].append(jarque_bera(sim_data)[0])
            
            if original_stats['shapiro'] is not None:
                simulation_stats['shapiro'].append(shapiro(sim_data)[0])
            
            simulation_stats['skewness'].append(abs(stats.skew(sim_data)))
            simulation_stats['kurtosis'].append(abs(stats.kurtosis(sim_data, fisher=True)))
        
        # Calcular p-values simulados
        results = {
            'n_simulations': n_simulations,
            'sample_size': n,
            'alpha': alpha,
            'tests': {}
        }
        
        for stat_name in simulation_stats.keys():
            if stat_name in original_stats and original_stats[stat_name] is not None:
                # P-value como proporção de simulações com estatística maior
                simulated_values = np.array(simulation_stats[stat_name])
                p_value_sim = np.mean(simulated_values >= original_stats[stat_name])
                
                results['tests'][stat_name] = {
                    'original_statistic': original_stats[stat_name],
                    'simulated_p_value': p_value_sim,
                    'is_normal': p_value_sim > alpha,
                    'simulated_mean': np.mean(simulated_values),
                    'simulated_std': np.std(simulated_values),
                    'percentile_rank': stats.percentileofscore(simulated_values, original_stats[stat_name])
                }
        
        # Consenso
        valid_tests = [test for test in results['tests'].values() if 'is_normal' in test]
        if valid_tests:
            normal_count = sum(1 for test in valid_tests if test['is_normal'])
            results['consensus'] = {
                'proportion_normal': normal_count / len(valid_tests),
                'is_normal': normal_count / len(valid_tests) > 0.5,
                'recommendation': "Normal" if normal_count / len(valid_tests) > 0.5 else "Não-Normal"
            }
        
        return results
    
    def parametric_vs_nonparametric_comparison(self, series1, series2):
        """
        Compara performance de testes paramétricos vs não-paramétricos
        
        Args:
            series1, series2 (pd.Series): Duas séries para comparação
            
        Returns:
            dict: Resultados da comparação
        """
        clean_data1 = series1.dropna()
        clean_data2 = series2.dropna()
        
        if len(clean_data1) < 10 or len(clean_data2) < 10:
            return {"error": "Dados insuficientes para comparação"}
        
        results = {
            'sample_sizes': {'series1': len(clean_data1), 'series2': len(clean_data2)},
            'parametric_tests': {},
            'nonparametric_tests': {}
        }
        
        # Testes paramétricos
        try:
            # T-test (assume normalidade)
            t_stat, t_p = stats.ttest_ind(clean_data1, clean_data2)
            results['parametric_tests']['t_test'] = {
                'statistic': t_stat,
                'p_value': t_p,
                'significant': t_p < 0.05,
                'assumes': 'Normalidade e homogeneidade de variâncias'
            }
            
            # Welch's t-test (não assume variâncias iguais)
            welch_stat, welch_p = stats.ttest_ind(clean_data1, clean_data2, equal_var=False)
            results['parametric_tests']['welch_t_test'] = {
                'statistic': welch_stat,
                'p_value': welch_p,
                'significant': welch_p < 0.05,
                'assumes': 'Normalidade (mas não variâncias iguais)'
            }
            
        except Exception as e:
            results['parametric_tests']['error'] = str(e)
        
        # Testes não-paramétricos
        try:
            # Mann-Whitney U
            mw_stat, mw_p = stats.mannwhitneyu(clean_data1, clean_data2, alternative='two-sided')
            results['nonparametric_tests']['mann_whitney'] = {
                'statistic': mw_stat,
                'p_value': mw_p,
                'significant': mw_p < 0.05,
                'assumes': 'Apenas que distribuições são contínuas'
            }
            
            # Wilcoxon rank-sum (alternativa ao Mann-Whitney)
            ws_stat, ws_p = stats.ranksums(clean_data1, clean_data2)
            results['nonparametric_tests']['wilcoxon_ranksum'] = {
                'statistic': ws_stat,
                'p_value': ws_p,
                'significant': ws_p < 0.05,
                'assumes': 'Distribuições contínuas'
            }
            
            # Kolmogorov-Smirnov para duas amostras
            ks2_stat, ks2_p = stats.ks_2samp(clean_data1, clean_data2)
            results['nonparametric_tests']['kolmogorov_smirnov_2samp'] = {
                'statistic': ks2_stat,
                'p_value': ks2_p,
                'significant': ks2_p < 0.05,
                'assumes': 'Distribuições contínuas'
            }
            
        except Exception as e:
            results['nonparametric_tests']['error'] = str(e)
        
        # Análise de consistência
        parametric_significant = []
        nonparametric_significant = []
        
        for test in results['parametric_tests'].values():
            if isinstance(test, dict) and 'significant' in test:
                parametric_significant.append(test['significant'])
        
        for test in results['nonparametric_tests'].values():
            if isinstance(test, dict) and 'significant' in test:
                nonparametric_significant.append(test['significant'])
        
        if parametric_significant and nonparametric_significant:
            param_agree = np.mean(parametric_significant)
            nonparam_agree = np.mean(nonparametric_significant)
            
            results['consistency_analysis'] = {
                'parametric_agreement': param_agree,
                'nonparametric_agreement': nonparam_agree,
                'methods_agree': abs(param_agree - nonparam_agree) < 0.5,
                'recommendation': self._get_test_recommendation(
                    results['parametric_tests'], 
                    results['nonparametric_tests'],
                    clean_data1, clean_data2
                )
            }
        
        return results
    
    def _get_test_recommendation(self, param_tests, nonparam_tests, data1, data2):
        """
        Gera recomendação sobre qual tipo de teste usar
        
        Args:
            param_tests, nonparam_tests (dict): Resultados dos testes
            data1, data2 (pd.Series): Dados originais
            
        Returns:
            str: Recomendação
        """
        # Verificar normalidade dos dados
        try:
            _, p1_norm = shapiro(data1) if len(data1) <= 5000 else (None, None)
            _, p2_norm = shapiro(data2) if len(data2) <= 5000 else (None, None)
            
            both_normal = (p1_norm and p1_norm > 0.05) and (p2_norm and p2_norm > 0.05)
            
            if both_normal:
                return "Usar testes paramétricos - dados seguem normalidade"
            else:
                return "Usar testes não-paramétricos - dados não seguem normalidade"
                
        except Exception:
            return "Usar testes não-paramétricos por segurança"
    
    def visual_normality_assessment(self, series, title="Análise Visual de Normalidade"):
        """
        Cria visualizações para avaliação de normalidade
        
        Args:
            series (pd.Series): Série temporal
            title (str): Título dos gráficos
            
        Returns:
            dict: Figuras para análise visual
        """
        clean_data = series.dropna()
        
        if len(clean_data) < 10:
            return {"error": "Dados insuficientes para análise visual"}
        
        # Criar subplot com múltiplas visualizações
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Histograma vs Normal',
                'Q-Q Plot vs Normal',
                'Box Plot',
                'Densidade vs Normal'
            ),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": True}]]
        )
        
        # 1. Histograma com curva normal
        hist_values, bin_edges = np.histogram(clean_data, bins=30, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        fig.add_trace(
            go.Bar(x=bin_centers, y=hist_values, name='Histograma', opacity=0.7),
            row=1, col=1
        )
        
        # Curva normal teórica
        x_normal = np.linspace(clean_data.min(), clean_data.max(), 100)
        y_normal = stats.norm.pdf(x_normal, clean_data.mean(), clean_data.std())
        
        fig.add_trace(
            go.Scatter(x=x_normal, y=y_normal, mode='lines', name='Normal Teórica'),
            row=1, col=1
        )
        
        # 2. Q-Q Plot
        theoretical_quantiles, sample_quantiles = probplot(clean_data, dist='norm', plot=None)
        
        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles, 
                y=sample_quantiles, 
                mode='markers', 
                name='Q-Q Points',
                marker=dict(size=4)
            ),
            row=1, col=2
        )
        
        # Linha de referência para Q-Q
        min_val = min(min(theoretical_quantiles), min(sample_quantiles))
        max_val = max(max(theoretical_quantiles), max(sample_quantiles))
        
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val], 
                y=[min_val, max_val], 
                mode='lines', 
                name='Linha Referência',
                line=dict(color='red', dash='dash')
            ),
            row=1, col=2
        )
        
        # 3. Box Plot
        fig.add_trace(
            go.Box(y=clean_data, name='Box Plot', boxpoints='outliers'),
            row=2, col=1
        )
        
        # 4. Densidade estimada vs Normal
        from scipy.stats import gaussian_kde
        
        # Densidade empírica
        kde = gaussian_kde(clean_data)
        x_density = np.linspace(clean_data.min(), clean_data.max(), 100)
        density_emp = kde(x_density)
        
        fig.add_trace(
            go.Scatter(x=x_density, y=density_emp, mode='lines', name='Densidade Empírica'),
            row=2, col=2
        )
        
        # Densidade normal teórica
        density_norm = stats.norm.pdf(x_density, clean_data.mean(), clean_data.std())
        
        fig.add_trace(
            go.Scatter(x=x_density, y=density_norm, mode='lines', name='Densidade Normal'),
            row=2, col=2
        )
        
        fig.update_layout(
            title=title,
            showlegend=True,
            height=800,
            width=1000
        )
        
        # Estatísticas resumo
        statistics = {
            'mean': clean_data.mean(),
            'median': clean_data.median(),
            'std': clean_data.std(),
            'skewness': stats.skew(clean_data),
            'kurtosis': stats.kurtosis(clean_data, fisher=True),
            'min': clean_data.min(),
            'max': clean_data.max(),
            'q25': clean_data.quantile(0.25),
            'q75': clean_data.quantile(0.75),
            'iqr': clean_data.quantile(0.75) - clean_data.quantile(0.25)
        }
        
        # Interpretação visual
        interpretation = self._interpret_visual_normality(statistics)
        
        return {
            'figure': fig,
            'statistics': statistics,
            'interpretation': interpretation,
            'sample_size': len(clean_data)
        }
    
    def _interpret_visual_normality(self, stats):
        """
        Interpreta estatísticas para avaliação visual de normalidade
        
        Args:
            stats (dict): Estatísticas descritivas
            
        Returns:
            dict: Interpretação das estatísticas
        """
        interpretation = {
            'central_tendency': {},
            'symmetry': {},
            'tail_behavior': {},
            'overall_assessment': ''
        }
        
        # Tendência central
        mean_median_diff = abs(stats['mean'] - stats['median'])
        relative_diff = mean_median_diff / stats['std'] if stats['std'] > 0 else 0
        
        interpretation['central_tendency'] = {
            'mean_median_difference': mean_median_diff,
            'relative_difference': relative_diff,
            'assessment': 'Simétrica' if relative_diff < 0.1 else 'Assimétrica'
        }
        
        # Simetria
        skew_val = stats['skewness']
        if abs(skew_val) < 0.5:
            skew_assessment = 'Aproximadamente simétrica'
        elif abs(skew_val) < 1.0:
            skew_assessment = 'Moderadamente assimétrica'
        else:
            skew_assessment = 'Altamente assimétrica'
        
        interpretation['symmetry'] = {
            'skewness_value': skew_val,
            'assessment': skew_assessment,
            'direction': 'Direita' if skew_val > 0 else 'Esquerda' if skew_val < 0 else 'Nenhuma'
        }
        
        # Comportamento das caudas
        kurt_val = stats['kurtosis']
        if abs(kurt_val) < 0.5:
            kurt_assessment = 'Caudas normais (mesocúrtica)'
        elif kurt_val > 0.5:
            kurt_assessment = 'Caudas pesadas (leptocúrtica)'
        else:
            kurt_assessment = 'Caudas leves (platicúrtica)'
        
        interpretation['tail_behavior'] = {
            'kurtosis_value': kurt_val,
            'assessment': kurt_assessment
        }
        
        # Avaliação geral
        normality_score = 0
        
        if relative_diff < 0.1:
            normality_score += 1
        if abs(skew_val) < 0.5:
            normality_score += 1
        if abs(kurt_val) < 0.5:
            normality_score += 1
        
        if normality_score == 3:
            overall = 'Dados aparentam seguir distribuição normal'
        elif normality_score == 2:
            overall = 'Dados são razoavelmente próximos à normalidade'
        elif normality_score == 1:
            overall = 'Dados mostram desvios moderados da normalidade'
        else:
            overall = 'Dados não aparentam seguir distribuição normal'
        
        interpretation['overall_assessment'] = overall
        
        return interpretation
