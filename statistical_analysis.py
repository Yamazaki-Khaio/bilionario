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
                try:
                    difference_score = (
                        (1 - comparison['comparison_tests']['ks_test']['p_value']) * 0.3 +
                        (1 - comparison['comparison_tests']['mann_whitney']['p_value']) * 0.2 +
                        abs(comparison['statistics']['asset1']['skewness'] - 
                            comparison['statistics']['asset2']['skewness']) * 0.2 +
                        abs(comparison['statistics']['asset1']['kurtosis'] - 
                            comparison['statistics']['asset2']['kurtosis']) * 0.3
                    )
                except KeyError as e:
                    raise Exception(f"Erro ao comparar distribuições: {str(e)}")
                
                
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
        # Verificar tamanho mínimo para executar testes estatísticos
        min_observations = 30  # Normalmente se considera 30 como mínimo para testes paramétricos

        if len(data1.values) < min_observations or len(data2.values) < min_observations:
            # Se não houver observações suficientes, definir valores padrão
            ks_stat, ks_p = 0, 1.0
            mw_stat, mw_p = 0, 1.0
        else:
            try:
                # Teste de Kolmogorov-Smirnov (distribuições diferentes)
                ks_stat, ks_p = stats.ks_2samp(data1.values, data2.values)
                
                # Teste Mann-Whitney U (medianas diferentes)
                mw_stat, mw_p = stats.mannwhitneyu(data1.values, data2.values, alternative='two-sided')
            except Exception as e:
                # Em caso de erro, definir valores padrão
                ks_stat, ks_p = 0, 1.0
                mw_stat, mw_p = 0, 1.0
          # Se houver observações suficientes, realizar os testes de Levene e t-test
        if len(data1.values) >= min_observations and len(data2.values) >= min_observations:
            try:
                # Teste Levene (variâncias diferentes)
                levene_stat, levene_p = stats.levene(data1.values, data2.values)
                
                # Teste t para médias (assumindo normalidade)
                t_stat, t_p = stats.ttest_ind(data1.values, data2.values)
            except Exception as e:
                # Em caso de erro, definir valores padrão
                levene_stat, levene_p = 0, 1.0
                t_stat, t_p = 0, 1.0
        else:
            # Valores padrão para poucos dados
            levene_stat, levene_p = 0, 1.0
            t_stat, t_p = 0, 1.0
        
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

class BacktestEngine:
    """Engine de backtesting baseado no bt library style"""
    
    def __init__(self, data):
        self.data = data
        self.returns = data.pct_change().dropna()
        
    def run_strategy_backtest(self, strategy_weights, start_date=None, end_date=None):
        """
        Executa backtest de uma estratégia
        
        Args:
            strategy_weights (dict): Pesos dos ativos
            start_date, end_date: Período do backtest
            
        Returns:
            dict: Resultados do backtest
        """
        # Filtrar período
        if start_date:
            returns_period = self.returns[self.returns.index >= start_date]
        else:
            returns_period = self.returns
            
        if end_date:
            returns_period = returns_period[returns_period.index <= end_date]
        
        # Calcular retornos da estratégia
        strategy_returns = self._calculate_strategy_returns(returns_period, strategy_weights)
        
        # Calcular métricas
        metrics = self._calculate_performance_metrics(strategy_returns)
        
        # Criar curva de equity
        equity_curve = (1 + strategy_returns).cumprod()
        
        return {
            'returns': strategy_returns,
            'equity_curve': equity_curve,
            'metrics': metrics,
            'period': {'start': returns_period.index[0], 'end': returns_period.index[-1]}
        }
    
    def _calculate_strategy_returns(self, returns, weights):
        """Calcula retornos da estratégia com rebalanceamento"""
        # Normalizar pesos
        total_weight = sum(weights.values())
        normalized_weights = {k: v/total_weight for k, v in weights.items()}
        
        # Calcular retornos ponderados
        strategy_returns = pd.Series(0.0, index=returns.index)
        
        for asset, weight in normalized_weights.items():
            if asset in returns.columns:
                strategy_returns += returns[asset] * weight
        
        return strategy_returns.fillna(0)
    
    def _calculate_performance_metrics(self, returns):
        """Calcula métricas de performance estilo bt library"""
        # Métricas básicas
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + returns).prod() ** (252/len(returns)) - 1
        annual_vol = returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        risk_free_rate = 0.10  # Selic aproximada
        sharpe = (annual_return - risk_free_rate) / annual_vol if annual_vol > 0 else 0
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino = (annual_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0
        
        # Max Drawdown
        equity_curve = (1 + returns).cumprod()
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve / running_max - 1)
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Métricas de período
        mtd = self._calculate_period_return(returns, 'M')
        qtd = self._calculate_period_return(returns, 'Q')
        ytd = self._calculate_period_return(returns, 'Y')
        
        # Métricas de distribuição
        daily_skew = skew(returns.dropna())
        daily_kurt = kurtosis(returns.dropna(), fisher=True)
        best_day = returns.max()
        worst_day = returns.min()
        
        # Análise mensal
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        monthly_sharpe = monthly_returns.mean() / monthly_returns.std() * np.sqrt(12) if monthly_returns.std() > 0 else 0
        monthly_vol = monthly_returns.std() * np.sqrt(12)
        
        # Win rates
        win_days = (returns > 0).sum() / len(returns) * 100
        win_months = (monthly_returns > 0).sum() / len(monthly_returns) * 100
        
        return {
            'start': returns.index[0],
            'end': returns.index[-1],
            'risk_free_rate': risk_free_rate,
            'total_return': total_return,
            'daily_sharpe': sharpe,
            'daily_sortino': sortino,
            'cagr': annual_return,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar,
            'mtd': mtd,
            'qtd': qtd,
            'ytd': ytd,
            'daily_mean_ann': annual_return,
            'daily_vol_ann': annual_vol,
            'daily_skew': daily_skew,
            'daily_kurt': daily_kurt,
            'best_day': best_day,
            'worst_day': worst_day,
            'monthly_sharpe': monthly_sharpe,
            'monthly_vol': monthly_vol,
            'monthly_mean_ann': monthly_returns.mean() * 12,
            'monthly_skew': skew(monthly_returns.dropna()),
            'monthly_kurt': kurtosis(monthly_returns.dropna(), fisher=True),
            'best_month': monthly_returns.max(),
            'worst_month': monthly_returns.min(),
            'win_rate_daily': win_days,
            'win_rate_monthly': win_months,
            'avg_up_month': monthly_returns[monthly_returns > 0].mean(),
            'avg_down_month': monthly_returns[monthly_returns < 0].mean()
        }
    
    def _calculate_period_return(self, returns, period):
        """Calcula retorno para período específico"""
        try:
            if period == 'M':  # Month to date
                current_month = returns.index[-1].replace(day=1)
                period_returns = returns[returns.index >= current_month]
            elif period == 'Q':  # Quarter to date
                current_quarter = returns.index[-1].replace(month=((returns.index[-1].month-1)//3)*3+1, day=1)
                period_returns = returns[returns.index >= current_quarter]
            elif period == 'Y':  # Year to date
                current_year = returns.index[-1].replace(month=1, day=1)
                period_returns = returns[returns.index >= current_year]
            else:
                return 0
                
            return (1 + period_returns).prod() - 1
        except Exception:
            return 0
    
    def plot_histogram(self, returns, title="Return Distribution"):
        """Plota histograma dos retornos estilo bt library"""
        fig = go.Figure()
          # Histograma
        fig.add_trace(go.Histogram(
            x=returns,
            nbinsx=50,
            opacity=0.7,
            name='Returns',
            histnorm=PROBABILITY_DENSITY
        ))
        
        # Curva normal para comparação
        mean = returns.mean()
        std = returns.std()
        x_norm = np.linspace(returns.min(), returns.max(), 100)
        y_norm = stats.norm.pdf(x_norm, mean, std)
        
        fig.add_trace(go.Scatter(
            x=x_norm,
            y=y_norm,
            mode='lines',
            name='Normal Distribution',
            line=dict(color='red', dash='dash')
        ))
        
        # Curva t-Student para comparação
        try:
            t_params = stats.t.fit(returns)
            y_t = stats.t.pdf(x_norm, *t_params)
            fig.add_trace(go.Scatter(
                x=x_norm,
                y=y_t,
                mode='lines',
                name='t-Student Distribution',
                line=dict(color='green', dash='dot')            ))
        except Exception:
            pass
        
        fig.update_layout(
            title=title,
            xaxis_title='Returns',
            yaxis_title='Density',
            showlegend=True
        )
        
        return fig
    
    def display_performance_table(self, metrics):
        """Exibe tabela de performance estilo bt library"""
        performance_data = {
            'Stat': [
                'Start', 'End', 'Risk-free rate',
                'Total Return', 'Daily Sharpe', 'Daily Sortino', 'CAGR',
                'Max Drawdown', 'Calmar Ratio',
                'MTD', 'QTD', 'YTD',
                'Daily Mean (ann.)', 'Daily Vol (ann.)', 'Daily Skew', 'Daily Kurt',
                'Best Day', 'Worst Day',
                'Monthly Sharpe', 'Monthly Vol (ann.)', 'Monthly Mean (ann.)',
                'Monthly Skew', 'Monthly Kurt', 'Best Month', 'Worst Month',
                'Win Rate Daily %', 'Win Rate Monthly %', 'Avg. Up Month', 'Avg. Down Month'
            ],
            'Value': [
                metrics['start'].strftime('%Y-%m-%d'),
                metrics['end'].strftime('%Y-%m-%d'),
                f"{metrics['risk_free_rate']:.2%}",
                f"{metrics['total_return']:.2%}",
                f"{metrics['daily_sharpe']:.2f}",
                f"{metrics['daily_sortino']:.2f}",
                f"{metrics['cagr']:.2%}",
                f"{metrics['max_drawdown']:.2%}",
                f"{metrics['calmar_ratio']:.2f}",
                f"{metrics['mtd']:.2%}",
                f"{metrics['qtd']:.2%}",
                f"{metrics['ytd']:.2%}",
                f"{metrics['daily_mean_ann']:.2%}",
                f"{metrics['daily_vol_ann']:.2%}",
                f"{metrics['daily_skew']:.2f}",
                f"{metrics['daily_kurt']:.2f}",
                f"{metrics['best_day']:.2%}",
                f"{metrics['worst_day']:.2%}",
                f"{metrics['monthly_sharpe']:.2f}",
                f"{metrics['monthly_vol']:.2%}",
                f"{metrics['monthly_mean_ann']:.2%}",
                f"{metrics['monthly_skew']:.2f}",
                f"{metrics['monthly_kurt']:.2f}",
                f"{metrics['best_month']:.2%}",
                f"{metrics['worst_month']:.2%}",
                f"{metrics['win_rate_daily']:.1f}%",
                f"{metrics['win_rate_monthly']:.1f}%",
                f"{metrics['avg_up_month']:.2%}",
                f"{metrics['avg_down_month']:.2%}"
            ]
        }
        
        return pd.DataFrame(performance_data)

    def create_risk_profile(self):
        """
        Cria um perfil de risco detalhado do ativo
        
        Returns:
            dict: Perfil de risco com métricas, classificação e recomendações
        """
        if self.risk_metrics is None:
            self.analyze_risk()
        
        vol_anual = self.risk_metrics['vol_annual']
        max_dd = abs(self.risk_metrics['max_drawdown'])
        sharpe = self.risk_metrics['sharpe_ratio']
        
        # Calcular score de risco
        risk_score = self._calculate_risk_score(vol_anual, max_dd, sharpe)
        
        # Determinar classificação de risco
        classification = self._get_risk_classification(risk_score)
        
        # Gerar alertas específicos
        alerts = self._generate_risk_alerts(vol_anual, max_dd, sharpe, self.risk_metrics)
        
        # Gerar estratégias de mitigação
        strategies = self._generate_mitigation_strategies(vol_anual, max_dd, sharpe)
        
        return {
            'risk_score': risk_score,
            'classification_level': classification['level'],
            'color': classification['color'],
            'recommendation': classification['recommendation'],
            'suggestions': classification['suggestions'],
            'specific_alerts': alerts,
            'mitigation_strategies': strategies,
            'metrics': {
                'annual_volatility': vol_anual,
                'max_drawdown_abs': max_dd,
                'sharpe_ratio': sharpe
            }        
        }
    
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
    
    def extreme_analysis_any_asset(self, asset_symbol, threshold=0.10):
        """
        Análise de extremos para qualquer ativo
        
        Args:
            asset_symbol (str): Símbolo do ativo
            threshold (float): Threshold para queda (0.10 = 10%)
            
        Returns:
            dict: Resultados da análise
        """
        if not hasattr(self, 'returns') or asset_symbol not in getattr(self, 'returns', {}).columns:
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
        try:
            extreme_analysis = self.extreme_analysis_any_asset(asset_symbol, threshold)
        except Exception:
            # Se não conseguir usar o método específico, criar uma análise básica
            extreme_analysis = {
                "probability": len(asset_returns[asset_returns <= -threshold]) / len(asset_returns)
            }
        
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
            'risk_metrics': risk_metrics
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

    def compare_distributions(self, asset1, asset2, min_observations=None):
        """
        Compara as distribuições de dois ativos
        
        Args:
            asset1, asset2: Nomes dos ativos
            min_observations: Número mínimo de observações (opcional)
            
        Returns:
            dict: Estatísticas e resultados da comparação
        """
        if min_observations is None:
            min_observations = self.min_observations
            
        # Verificar se há dados suficientes antes de comparar
        if asset1 not in self.returns.columns or asset2 not in self.returns.columns:
            return {"error": f"Ativos {asset1} ou {asset2} não encontrados"}
            
        data1 = self.returns[asset1].dropna()
        data2 = self.returns[asset2].dropna()
        
        if len(data1) < min_observations or len(data2) < min_observations:
            return {"error": f"Dados insuficientes para análise. Mínimo: {min_observations} observações"}
            
        # Alinhar séries temporalmente
        aligned_data = pd.concat([data1, data2], axis=1).dropna()
        data1 = aligned_data.iloc[:, 0]
        data2 = aligned_data.iloc[:, 1]
        
        if len(aligned_data) < min_observations:
            return {"error": f"Dados alinhados insuficientes ({len(aligned_data)}) para análise. Mínimo: {min_observations} observações"}
        
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
        # Verificar tamanho mínimo para executar testes estatísticos 
        min_observations = 30  # Normalmente se considera 30 como mínimo para testes paramétricos
        
        if len(data1.values) < min_observations or len(data2.values) < min_observations:
            # Se não houver observações suficientes, definir valores padrão
            ks_stat, ks_p = 0, 1.0
            mw_stat, mw_p = 0, 1.0
        else:
            try:
                # Teste de Kolmogorov-Smirnov (distribuições diferentes)
                ks_stat, ks_p = stats.ks_2samp(data1.values, data2.values)
                
                # Teste Mann-Whitney U (medianas diferentes)
                mw_stat, mw_p = stats.mannwhitneyu(data1.values, data2.values, alternative='two-sided')
            except Exception as e:
                # Em caso de erro, definir valores padrão
                ks_stat, ks_p = 0, 1.0
                mw_stat, mw_p = 0, 1.0
          # Se houver observações suficientes, realizar os testes de Levene e t-test
        if len(data1.values) >= min_observations and len(data2.values) >= min_observations:
            try:
                # Teste Levene (variâncias diferentes)
                levene_stat, levene_p = stats.levene(data1.values, data2.values)
                
                # Teste t para médias (assumindo normalidade)
                t_stat, t_p = stats.ttest_ind(data1.values, data2.values)
            except Exception as e:
                # Em caso de erro, definir valores padrão
                levene_stat, levene_p = 0, 1.0
                t_stat, t_p = 0, 1.0
        else:
            # Valores padrão para poucos dados
            levene_stat, levene_p = 0, 1.0
            t_stat, t_p = 0, 1.0
        
        return {
            'assets': {'asset1': asset1, 'asset2': asset2},
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
        
    def petrobras_extreme_analysis(self, threshold=0.10):
        """
        Mantida para compatibilidade, agora usa a função genérica asset_extreme_analysis
        """
        return self.asset_extreme_analysis(PETR4_SYMBOL, threshold)
        
    def asset_extreme_analysis(self, asset_symbol, threshold=0.10):
        """
        Realiza análise de eventos extremos para qualquer ativo
        
        Esta função analisa comportamentos extremos dos preços do ativo escolhido,
        calculando estatísticas sobre quedas superiores ao threshold e padrões de recuperação.
        Metodologia baseada nos estudos de análise de eventos extremos.
        
        Args:
            asset_symbol (str): Símbolo do ativo para análise
            threshold (float): Limite para considerar um evento como extremo (ex: 0.10 = 10%)
            
        Returns:
            dict: Resultados da análise de eventos extremos
        """
        if asset_symbol not in self.returns.columns:
            return {"error": f"Ativo {asset_symbol} não encontrado nos dados"}
            
        asset_returns = self.returns[asset_symbol].dropna()
        
        if len(asset_returns) < 100:
            return {"error": "Dados insuficientes para análise de eventos extremos"}
        
        # Identificar dias com quedas extremas (abaixo do threshold negativo)
        extreme_falls = asset_returns[asset_returns <= -threshold]
        
        # Calcular estatísticas diárias
        daily_stats = {
            'mean': asset_returns.mean(),
            'std': asset_returns.std(),
            'skewness': skew(asset_returns.dropna()),
            'kurtosis': kurtosis(asset_returns.dropna(), fisher=True),  # Excesso de curtose
            'min': asset_returns.min(),
            'max': asset_returns.max()
        }
        
        # Calcular probabilidade empírica de eventos extremos
        empirical_probability = len(extreme_falls) / len(asset_returns)
        
        # Estatísticas dos eventos extremos
        extreme_stats = {
            'mean': extreme_falls.mean() if len(extreme_falls) > 0 else None,
            'std': extreme_falls.std() if len(extreme_falls) > 0 else None,
            'worst_fall': extreme_falls.min() if len(extreme_falls) > 0 else None,
            'dates': extreme_falls.index.tolist()
        }
        
        # Análise de recuperação após eventos extremos
        recovery_periods = []
        recovery_stats = {}
        
        if len(extreme_falls) > 0:
            for date in extreme_falls.index:
                if date < self.data.index[-30]:  # Garantir que há pelo menos 30 dias após o evento
                    idx = self.data.index.get_loc(date)
                    future_prices = self.data.iloc[idx:idx+31, self.data.columns.get_loc(asset_symbol)]
                    
                    # Calcular dias até recuperação
                    base_price = future_prices.iloc[0]
                    recovery_day = None
                    
                    for i, (date_future, price) in enumerate(future_prices.items()):
                        if price >= base_price and i > 0:
                            recovery_day = i
                            break
                    
                    if recovery_day:
                        recovery_periods.append(recovery_day)
            
            if recovery_periods:
                recovery_stats = {
                    'mean_days': np.mean(recovery_periods),
                    'median_days': np.median(recovery_periods),
                    'max_days': max(recovery_periods),
                    'recovery_rate': len([x for x in recovery_periods if x <= 30]) / len(recovery_periods)
                }
        
        return {
            'asset_symbol': asset_symbol,
            'total_days': len(asset_returns),
            'extreme_falls_count': len(extreme_falls),
            'probability': empirical_probability,
            'daily_statistics': daily_stats,
            'extreme_statistics': extreme_stats,
            'recovery_statistics': recovery_stats
        }
