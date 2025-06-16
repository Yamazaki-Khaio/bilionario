# statistical_analysis.py
"""
Sistema de Análise Estatística Avançada
Implementa análises de distribuições, testes estatísticos e análise de risco
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import shapiro, jarque_bera, kstest, anderson, normaltest
from scipy.stats import t, skew, kurtosis, percentileofscore
from financial_formatting import format_percentage, format_ratio
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# Constantes
PETR4_SYMBOL = 'PETR4.SA'
PROBABILITY_DENSITY = 'probability density'

class StatisticalAnalysis:
    """Classe para análises estatísticas avançadas de ativos financeiros"""
    
    def __init__(self, data):
        """
        Inicializa a análise estatística
        
        Args:
            data (pd.DataFrame): DataFrame com preços dos ativos
        """
        self.data = data
        self.returns = data.pct_change().dropna()
        
    def petrobras_extreme_analysis(self, threshold=0.10):
        """
        Análise específica da Petrobras para quedas superiores a threshold
        
        Args:
            threshold (float): Threshold para queda (0.10 = 10%)
            
        Returns:
            dict: Resultados da análise
        """
        if PETR4_SYMBOL not in self.returns.columns:
            return {"error": "Dados da Petrobras não encontrados"}
            
        petr_returns = self.returns[PETR4_SYMBOL].dropna()
        
        # Análise de quedas extremas
        extreme_falls = petr_returns[petr_returns <= -threshold]
        
        # Análise estatística
        results = {
            'total_days': len(petr_returns),
            'extreme_falls_count': len(extreme_falls),
            'probability': len(extreme_falls) / len(petr_returns),
            'extreme_falls_dates': extreme_falls.index.tolist(),
            'extreme_falls_values': extreme_falls.values.tolist(),
            'daily_statistics': {
                'mean': petr_returns.mean(),
                'std': petr_returns.std(),
                'skewness': skew(petr_returns),
                'kurtosis': kurtosis(petr_returns, fisher=True),
                'min': petr_returns.min(),
                'max': petr_returns.max()
            }
        }
        
        # Teste de normalidade
        shapiro_stat, shapiro_p = shapiro(petr_returns.values)
        jb_stat, jb_p = jarque_bera(petr_returns.values)
        
        results['normality_tests'] = {
            'shapiro': {'statistic': shapiro_stat, 'p_value': shapiro_p},
            'jarque_bera': {'statistic': jb_stat, 'p_value': jb_p},
            'is_normal': jb_p > 0.05
        }
        
        # Análise com distribuição t-Student
        if not results['normality_tests']['is_normal']:
            # Ajustar distribuição t-Student
            try:
                t_params = stats.t.fit(petr_returns.values)
                df_param = t_params[0]  # graus de liberdade
                loc_param = t_params[1]  # localização
                scale_param = t_params[2]  # escala
                
                # Probabilidade usando t-Student
                t_prob = stats.t.cdf(-threshold, df_param, loc_param, scale_param)
                
                results['t_student_analysis'] = {
                    'degrees_freedom': df_param,
                    'location': loc_param,
                    'scale': scale_param,
                    'probability_t_student': t_prob,
                    'recommended': True
                }
            except Exception:
                results['t_student_analysis'] = {'error': 'Falha ao ajustar t-Student'}
        
        # Análise com distribuição normal (para comparação)
        mean = petr_returns.mean()
        std = petr_returns.std()
        normal_prob = stats.norm.cdf(-threshold, mean, std)
        
        results['normal_analysis'] = {
            'probability_normal': normal_prob,
            'recommended': results['normality_tests']['is_normal']
        }
        
        return results
    
    def find_different_distributions(self, min_data_points=500):
        """
        Encontra dois ativos com distribuições estatisticamente diferentes
        
        Args:
            min_data_points (int): Mínimo de pontos de dados válidos
            
        Returns:
            dict: Resultados da análise
        """
        # Filtrar ativos com dados suficientes
        valid_assets = []
        for col in self.returns.columns:
            if self.returns[col].dropna().count() >= min_data_points:
                valid_assets.append(col)
        
        if len(valid_assets) < 2:
            return {"error": "Dados insuficientes para comparação"}
        
        best_comparison = None
        max_difference = 0
        
        # Comparar pares de ativos
        for i, asset1 in enumerate(valid_assets):
            for asset2 in valid_assets[i+1:]:
                
                data1 = self.returns[asset1].dropna()
                data2 = self.returns[asset2].dropna()
                
                # Alinhar dados temporalmente
                aligned_data = pd.concat([data1, data2], axis=1).dropna()
                if len(aligned_data) < min_data_points:
                    continue
                    
                aligned_data1 = aligned_data.iloc[:, 0]
                aligned_data2 = aligned_data.iloc[:, 1]
                
                # Testes estatísticos
                comparison = self._compare_distributions(
                    aligned_data1, aligned_data2, asset1, asset2
                )
                
                # Calcular diferença total (combinação de vários testes)
                difference_score = (
                    (1 - comparison['comparison_tests']['ks_test']['p_value']) * 0.3 +
                    (1 - comparison['comparison_tests']['mann_whitney']['p_value']) * 0.2 +
                    abs(comparison['statistics']['asset1']['skewness'] - 
                        comparison['statistics']['asset2']['skewness']) * 0.2 +
                    abs(comparison['statistics']['asset1']['kurtosis'] - 
                        comparison['statistics']['asset2']['kurtosis']) * 0.3
                )
                
                if difference_score > max_difference:
                    max_difference = difference_score
                    best_comparison = comparison
        
        return best_comparison
    
    def _compare_distributions(self, data1, data2, name1, name2):
        """
        Compara distribuições entre dois ativos
        
        Args:
            data1, data2: Séries de retornos
            name1, name2: Nomes dos ativos
            
        Returns:
            dict: Resultados detalhados da comparação
        """
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
        shapiro1 = shapiro(data1.values)
        shapiro2 = shapiro(data2.values)
        jb1 = jarque_bera(data1.values)
        jb2 = jarque_bera(data2.values)
        
        # Teste de Kolmogorov-Smirnov (distribuições diferentes)
        ks_stat, ks_p = stats.ks_2samp(data1.values, data2.values)
        
        # Teste Mann-Whitney U (medianas diferentes)
        mw_stat, mw_p = stats.mannwhitneyu(data1.values, data2.values, alternative='two-sided')
        
        # Teste Levene (variâncias diferentes)
        levene_stat, levene_p = stats.levene(data1.values, data2.values)
        
        # Teste t para médias (assumindo normalidade)
        t_stat, t_p = stats.ttest_ind(data1.values, data2.values)
        
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
    
    def create_distribution_comparison_plot(self, asset1, asset2):
        """
        Cria gráficos comparativos de distribuições
        
        Args:
            asset1, asset2: Nomes dos ativos
            
        Returns:
            plotly figure
        """
        if asset1 not in self.returns.columns or asset2 not in self.returns.columns:
            return None
            
        data1 = self.returns[asset1].dropna()
        data2 = self.returns[asset2].dropna()
        
        # Criar subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                f'Histograma - {asset1} vs {asset2}',
                'Q-Q Plot vs Normal',
                'Box Plot Comparativo',
                'Densidade Acumulada'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Histograma
        fig.add_trace(
            go.Histogram(x=data1, name=asset1, opacity=0.7, 
                        nbinsx=50, histnorm=PROBABILITY_DENSITY),
            row=1, col=1
        )
        fig.add_trace(
            go.Histogram(x=data2, name=asset2, opacity=0.7, 
                        nbinsx=50, histnorm=PROBABILITY_DENSITY),
            row=1, col=1
        )
        
        # 2. Q-Q Plot vs Normal
        (osm1, osr1), (slope1, intercept1, _) = stats.probplot(data1, dist="norm", plot=None)
        (osm2, osr2), (_, _, _) = stats.probplot(data2, dist="norm", plot=None)
        
        fig.add_trace(
            go.Scatter(x=osm1, y=osr1, mode='markers', name=f'{asset1} Q-Q',
                      marker=dict(color='blue')),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=osm2, y=osr2, mode='markers', name=f'{asset2} Q-Q',
                      marker=dict(color='red')),
            row=1, col=2
        )
        
        # Linha de referência normal
        fig.add_trace(
            go.Scatter(x=osm1, y=slope1 * osm1 + intercept1, 
                      mode='lines', name='Normal Ref', 
                      line=dict(color='gray', dash='dash')),
            row=1, col=2
        )
        
        # 3. Box Plot
        fig.add_trace(
            go.Box(y=data1, name=asset1, boxpoints='outliers'),
            row=2, col=1
        )
        fig.add_trace(
            go.Box(y=data2, name=asset2, boxpoints='outliers'),
            row=2, col=1
        )
        
        # 4. CDF Empírica
        x1_sorted = np.sort(data1)
        y1 = np.arange(1, len(x1_sorted) + 1) / len(x1_sorted)
        x2_sorted = np.sort(data2)
        y2 = np.arange(1, len(x2_sorted) + 1) / len(x2_sorted)
        
        fig.add_trace(
            go.Scatter(x=x1_sorted, y=y1, mode='lines', name=f'{asset1} CDF'),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=x2_sorted, y=y2, mode='lines', name=f'{asset2} CDF'),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text="Análise Comparativa de Distribuições",
            showlegend=True
        )
        
        return fig
    
    def create_petrobras_risk_analysis(self):
        """
        Análise específica de risco da Petrobras
        
        Returns:
            dict: Análise completa
        """
        if PETR4_SYMBOL not in self.returns.columns:
            return {"error": "Dados da Petrobras não encontrados"}
            
        petr_returns = self.returns[PETR4_SYMBOL].dropna()
        
        # VaR e CVaR
        var_95 = np.percentile(petr_returns, 5)
        var_99 = np.percentile(petr_returns, 1)
        cvar_95 = petr_returns[petr_returns <= var_95].mean()
        cvar_99 = petr_returns[petr_returns <= var_99].mean()
        
        # Análise de caudas
        tail_analysis = self._analyze_tails(petr_returns)
        
        # Análise de volatilidade
        volatility_analysis = self._analyze_volatility(petr_returns)
        
        return {
            'var_cvar': {
                'var_95': var_95,
                'var_99': var_99,
                'cvar_95': cvar_95,
                'cvar_99': cvar_99
            },
            'tail_analysis': tail_analysis,
            'volatility_analysis': volatility_analysis
        }
    
    def _analyze_tails(self, returns):
        """Análise das caudas da distribuição"""
        # Identificar valores extremos (além de 2 desvios padrão)
        mean = returns.mean()
        std = returns.std()
        
        left_tail = returns[returns < mean - 2*std]
        right_tail = returns[returns > mean + 2*std]
        
        return {
            'left_tail_count': len(left_tail),
            'right_tail_count': len(right_tail),
            'left_tail_freq': len(left_tail) / len(returns),
            'right_tail_freq': len(right_tail) / len(returns),
            'asymmetry': len(left_tail) - len(right_tail)
        }
    
    def _analyze_volatility(self, returns):
        """Análise de volatilidade"""
        # Volatilidade realizada (janelas móveis)
        vol_30d = returns.rolling(30).std() * np.sqrt(252)
        vol_60d = returns.rolling(60).std() * np.sqrt(252)
        
        # Clustering de volatilidade (GARCH-like)
        squared_returns = returns ** 2
        vol_clustering = squared_returns.rolling(5).mean().std()
        
        return {
            'current_vol_annual': returns.std() * np.sqrt(252),
            'vol_30d_current': vol_30d.iloc[-1] if not vol_30d.empty else None,
            'vol_60d_current': vol_60d.iloc[-1] if not vol_60d.empty else None,
            'vol_clustering_metric': vol_clustering,
            'max_vol_period': vol_30d.idxmax() if not vol_30d.empty else None,
            'min_vol_period': vol_30d.idxmin() if not vol_30d.empty else None
        }
    
    def extreme_analysis_any_asset(self, asset_symbol, threshold=0.10):
        """
        Análise de extremos para qualquer ativo
        
        Args:
            asset_symbol (str): Símbolo do ativo
            threshold (float): Threshold para queda (0.10 = 10%)
            
        Returns:
            dict: Resultados da análise
        """
        if asset_symbol not in self.returns.columns:
            return {"error": f"Dados do ativo {asset_symbol} não encontrados"}
            
        asset_returns = self.returns[asset_symbol].dropna()
        
        if len(asset_returns) < 20:
            return {"error": "Dados insuficientes para análise"}
        
        # Análise de quedas extremas
        extreme_falls = asset_returns[asset_returns <= -threshold]
        
        # Análise estatística
        results = {
            'asset_symbol': asset_symbol,
            'threshold': threshold,
            'total_days': len(asset_returns),
            'extreme_falls_count': len(extreme_falls),
            'probability': len(extreme_falls) / len(asset_returns),
            'extreme_falls_dates': extreme_falls.index.tolist(),
            'extreme_falls_values': extreme_falls.values.tolist(),
            'daily_statistics': {
                'mean': asset_returns.mean(),
                'std': asset_returns.std(),
                'skewness': skew(asset_returns),
                'kurtosis': kurtosis(asset_returns, fisher=True),
                'min': asset_returns.min(),
                'max': asset_returns.max()
            }
        }
        
        # Teste de normalidade
        try:
            shapiro_stat, shapiro_p = shapiro(asset_returns.values)
            jb_stat, jb_p = jarque_bera(asset_returns.values)
            
            results['normality_tests'] = {
                'shapiro': {'statistic': shapiro_stat, 'p_value': shapiro_p},
                'jarque_bera': {'statistic': jb_stat, 'p_value': jb_p},
                'is_normal': jb_p > 0.05
            }
        except Exception as e:
            results['normality_tests'] = {'error': f'Erro nos testes: {str(e)}'}
        
        # Análise com distribuição t-Student
        if 'normality_tests' in results and not results['normality_tests'].get('is_normal', True):
            try:
                t_params = stats.t.fit(asset_returns.values)
                df_param = t_params[0]  # graus de liberdade
                loc_param = t_params[1]  # localização
                scale_param = t_params[2]  # escala
                
                # Probabilidade usando t-Student
                t_prob = stats.t.cdf(-threshold, df_param, loc_param, scale_param)
                
                results['t_student_analysis'] = {
                    'degrees_freedom': df_param,
                    'location': loc_param,
                    'scale': scale_param,
                    'probability_t_student': t_prob,
                    'recommended': True
                }
            except Exception as e:
                results['t_student_analysis'] = {'error': f'Falha ao ajustar t-Student: {str(e)}'}
        
        # Análise com distribuição normal (para comparação)
        try:
            mean = asset_returns.mean()
            std = asset_returns.std()
            normal_prob = stats.norm.cdf(-threshold, mean, std)
            
            results['normal_analysis'] = {
                'probability_normal': normal_prob,
                'recommended': results.get('normality_tests', {}).get('is_normal', False)
            }
        except Exception as e:
            results['normal_analysis'] = {'error': f'Erro na análise normal: {str(e)}'}
        
        return results

    def create_generic_risk_analysis(self, asset_symbol, threshold=0.10):
        """
        Análise de risco genérica para qualquer ativo
        
        Args:
            asset_symbol (str): Símbolo do ativo para análise
            threshold (float): Limite para quedas extremas (padrão 10%)
            
        Returns:
            dict: Análise completa de risco do ativo
        """
        if asset_symbol not in self.returns.columns:
            return {"error": f"Dados do ativo {asset_symbol} não encontrados"}
        
        asset_returns = self.returns[asset_symbol].dropna()
        
        if len(asset_returns) < 20:
            return {"error": "Dados insuficientes para análise de risco"}
        
        # VaR e CVaR para o ativo
        var_95 = np.percentile(asset_returns, 5)
        var_99 = np.percentile(asset_returns, 1)
        cvar_95 = asset_returns[asset_returns <= var_95].mean()
        cvar_99 = asset_returns[asset_returns <= var_99].mean()
        
        # Análise de caudas genérica
        tail_analysis = self._analyze_tails(asset_returns)
        
        # Análise de volatilidade genérica
        volatility_analysis = self._analyze_volatility(asset_returns)
        
        # Análise de extremos específica
        extreme_analysis = self.extreme_analysis_any_asset(asset_symbol, threshold)
        
        # Métricas de risco adicionais
        risk_metrics = {
            'max_drawdown': self._calculate_max_drawdown(asset_returns),
            'downside_deviation': self._calculate_downside_deviation(asset_returns),
            'upside_potential': self._calculate_upside_potential(asset_returns),
            'risk_return_ratio': self._calculate_risk_return_ratio(asset_returns)
        }
        
        return {
            'asset_symbol': asset_symbol,
            'var_cvar': {
                'var_95': var_95,
                'var_99': var_99,
                'cvar_95': cvar_95,
                'cvar_99': cvar_99
            },
            'tail_analysis': tail_analysis,
            'volatility_analysis': volatility_analysis,
            'extreme_analysis': extreme_analysis,
            'risk_metrics': risk_metrics,
            'recommendation': self._generate_risk_recommendation(asset_returns, risk_metrics)
        }
    
    def _calculate_max_drawdown(self, returns):
        """Calcula o máximo drawdown de uma série de retornos"""
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        return drawdown.min()
    
    def _calculate_downside_deviation(self, returns):
        """Calcula o desvio padrão dos retornos negativos"""
        negative_returns = returns[returns < 0]
        return negative_returns.std() if len(negative_returns) > 0 else 0
    
    def _calculate_upside_potential(self, returns):
        """Calcula o potencial de alta baseado em retornos positivos"""
        positive_returns = returns[returns > 0]
        return positive_returns.mean() if len(positive_returns) > 0 else 0
    
    def _calculate_risk_return_ratio(self, returns):
        """Calcula a razão risco-retorno (Sharpe simplificado)"""
        mean_return = returns.mean()
        std_return = returns.std()
        return mean_return / std_return if std_return > 0 else 0
    
    def _generate_risk_recommendation(self, returns, risk_metrics):
        """Gera recomendação baseada na análise de risco"""
        vol_anual = returns.std() * np.sqrt(252)
        max_dd = abs(risk_metrics['max_drawdown'])
        sharpe = risk_metrics.get('sharpe_ratio', 0)
        
        # Calcular score de risco composto
        risk_score = self._calculate_risk_score(vol_anual, max_dd, sharpe)
        
        # Obter classificação e recomendações
        classification = self._get_risk_classification(risk_score)
        alerts = self._generate_risk_alerts(vol_anual, max_dd, sharpe, risk_metrics)
        strategies = self._generate_mitigation_strategies(vol_anual, max_dd, sharpe)
        
        return {
            'risk_level': classification['level'],
            'risk_score': round(risk_score, 2),
            'color': classification['color'],
            'recommendation': classification['recommendation'],
            'suggestions': classification['suggestions'],
            'specific_alerts': alerts,
            'mitigation_strategies': strategies,
            'metrics': {
                'annual_volatility': vol_anual,
                'max_drawdown_abs': max_dd,
                'sharpe_ratio': sharpe
            }        }
    
    def _calculate_risk_score(self, vol_anual, max_dd, sharpe):
        """Calcula score de risco composto"""
        vol_score = self._get_volatility_score(vol_anual)
        dd_score = self._get_drawdown_score(max_dd)
        sharpe_score = self._get_sharpe_score(sharpe)
        
        return vol_score * 0.4 + dd_score * 0.4 + sharpe_score * 0.2
        
    def _get_volatility_score(self, vol_anual):
        """Calcula score baseado na volatilidade"""
        if vol_anual < 0.15:
            return 1
        elif vol_anual < 0.25:
            return 2
        elif vol_anual < 0.35:
            return 3
        else:
            return 4
    
    def _get_drawdown_score(self, max_dd):
        """Calcula score baseado no drawdown"""
        if max_dd < 0.10:
            return 1
        elif max_dd < 0.20:
            return 2
        elif max_dd < 0.35:
            return 3
        else:
            return 4
    
    def _get_sharpe_score(self, sharpe):
        """Calcula score baseado no Sharpe Ratio"""
        if sharpe > 1.5:
            return 1
        elif sharpe > 1.0:
            return 2
        elif sharpe > 0.5:
            return 3
        else:
            return 4
    
    def _get_risk_classification(self, risk_score):
        """Retorna classificação baseada no score de risco"""
        classifications = {
            1.5: {
                'level': "MUITO BAIXO",
                'color': "🟢",
                'recommendation': "Ativo ideal para perfil conservador",
                'suggestions': ["Pode compor 60-80% da carteira", "Adequado para alocação principal"]
            },
            2.5: {
                'level': "BAIXO", 
                'color': "🟡",
                'recommendation': "Ativo adequado para perfil moderadamente conservador",
                'suggestions': ["Pode compor 40-60% do portfólio", "Adequado para diversificação"]
            },
            3.0: {
                'level': "MODERADO",
                'color': "🟠", 
                'recommendation': "Ativo de risco equilibrado",
                'suggestions': ["Limite a 30-40% do portfólio", "Implemente stop-loss em -15%"]
            },
            3.5: {
                'level': "ALTO",
                'color': "🔴",
                'recommendation': "Ativo de alto risco, apenas para perfil arrojado", 
                'suggestions': ["Limite a 15-25% do portfólio", "Stop-loss obrigatório"]
            }
        }
        
        for threshold, classification in classifications.items():
            if risk_score <= threshold:
                return classification
        
        return {
            'level': "MUITO ALTO",
            'color': "🚫",
            'recommendation': "Ativo de risco extremo",
            'suggestions': ["Máximo 5-10% do portfólio", "Monitoramento intraday necessário"]
        }
    
    def _generate_risk_alerts(self, vol_anual, max_dd, sharpe, risk_metrics):
        """Gera alertas específicos baseados nas métricas"""
        alerts = []
        
        if vol_anual > 0.40:
            alerts.append("⚠️ Volatilidade extrema detectada")
        if max_dd > 0.30:
            alerts.append("📉 Drawdown máximo elevado")
        if sharpe < 0:
            alerts.append("📊 Sharpe Ratio negativo")
        if risk_metrics.get('var_95', 0) < -0.05:
            alerts.append("💥 VaR 95% elevado")
            
        return alerts
        
    def _generate_mitigation_strategies(self, vol_anual, max_dd, sharpe):
        """Gera estratégias de mitigação de risco"""
        strategies = []
        
        if vol_anual > 0.25:
            strategies.append("Position sizing baseado na volatilidade")
        if max_dd > 0.20:
            strategies.append("Stop-loss baseado no drawdown histórico")
        if sharpe < 1.0:
            strategies.append("Combinar com ativos de maior Sharpe Ratio")
            
        return strategies
