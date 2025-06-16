# pair_trading_advanced.py
"""
Sistema Avançado de Pair Trading com Análise de Distribuições
Implementa estratégias baseadas em distribuições t-Student e normalização avançada
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import t, norm, jarque_bera, shapiro
from statsmodels.tsa.stattools import coint, adfuller
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import streamlit as st
import warnings
warnings.filterwarnings("ignore")

class PairTradingAdvanced:
    """Classe para análise avançada de pair trading com distribuições estatísticas"""
    
    def __init__(self, data):
        """
        Inicializa análise de pair trading
        
        Args:
            data (pd.DataFrame): DataFrame com preços dos ativos
        """
        self.data = data
        self.returns = data.pct_change().dropna()
        
    def analyze_pair_with_distributions(self, asset1, asset2, lookback_window=252):
        """
        Análise completa de um par com diferentes distribuições
        
        Args:
            asset1, asset2: Nomes dos ativos
            lookback_window: Janela para análise
            
        Returns:
            dict: Análise completa
        """
        # Obter dados alinhados
        prices1 = self.data[asset1].dropna()
        prices2 = self.data[asset2].dropna()
        
        # Alinhar temporalmente
        aligned_data = pd.concat([prices1, prices2], axis=1).dropna()
        if len(aligned_data) < lookback_window:
            return {"error": "Dados insuficientes"}
            
        aligned_prices1 = aligned_data.iloc[:, 0]
        aligned_prices2 = aligned_data.iloc[:, 1]
        
        # Teste de cointegração
        coint_result = self._test_cointegration_robust(aligned_prices1, aligned_prices2)
        
        if not coint_result['is_cointegrated']:
            return {"error": "Ativos não são cointegrados", "cointegration": coint_result}
        
        # Calcular spread
        spread_analysis = self._calculate_spread_with_distributions(
            aligned_prices1, aligned_prices2, coint_result['hedge_ratio']
        )
        
        # Análise de diferentes normalizações
        normalization_analysis = self._analyze_normalizations(spread_analysis['spread'])
        
        # Análise de distribuições
        distribution_analysis = self._analyze_spread_distributions(spread_analysis['spread'])
        
        # Sinais de trading
        trading_signals = self._generate_advanced_signals(
            spread_analysis, distribution_analysis, normalization_analysis
        )
        
        return {
            'assets': {'asset1': asset1, 'asset2': asset2},
            'cointegration': coint_result,
            'spread_analysis': spread_analysis,
            'normalization_analysis': normalization_analysis,
            'distribution_analysis': distribution_analysis,
            'trading_signals': trading_signals
        }
    
    def _test_cointegration_robust(self, prices1, prices2):
        """Teste robusto de cointegração com múltiples métodos"""
        try:
            # Teste Engle-Granger
            coint_stat, p_value, _ = coint(prices1, prices2)
            
            # Regressão para hedge ratio
            x = prices2.values.reshape(-1, 1)
            y = prices1.values
            
            # OLS simples
            hedge_ratio = np.linalg.lstsq(x, y, rcond=None)[0][0]
            
            # Teste de estacionariedade do spread
            spread = prices1 - hedge_ratio * prices2
            adf_stat, adf_p, _, _, adf_critical, _ = adfuller(spread, regression='c')
            
            # Análise de correlação
            correlation = prices1.corr(prices2)
            
            return {
                'is_cointegrated': p_value < 0.05,
                'coint_pvalue': p_value,
                'coint_statistic': coint_stat,
                'hedge_ratio': hedge_ratio,
                'adf_pvalue': adf_p,
                'adf_statistic': adf_stat,
                'adf_critical_values': adf_critical,
                'correlation': correlation,
                'spread_mean': spread.mean(),
                'spread_std': spread.std()
            }
        except Exception as e:
            return {
                'is_cointegrated': False,
                'error': str(e)
            }
    
    def _calculate_spread_with_distributions(self, prices1, prices2, hedge_ratio):
        """Calcula spread e analisa suas propriedades estatísticas"""
        # Spread básico
        spread = prices1 - hedge_ratio * prices2
        
        # Log spread (para casos com tendência)
        log_spread = np.log(prices1) - hedge_ratio * np.log(prices2)
        
        # Spread normalizado
        spread_normalized = (spread - spread.mean()) / spread.std()
        
        # Análise de rolling statistics
        rolling_mean = spread.rolling(window=30).mean()
        rolling_std = spread.rolling(window=30).std()
        rolling_z_score = (spread - rolling_mean) / rolling_std
        
        # Half-life do spread (velocidade de reversão)
        half_life = self._calculate_half_life(spread)
        
        return {
            'spread': spread,
            'log_spread': log_spread,
            'spread_normalized': spread_normalized,
            'rolling_z_score': rolling_z_score,
            'rolling_mean': rolling_mean,
            'rolling_std': rolling_std,
            'half_life': half_life,
            'hedge_ratio': hedge_ratio
        }
    
    def _calculate_half_life(self, spread):
        """Calcula half-life do spread para reversão à média"""
        try:
            # Regressão AR(1): spread[t] = alpha + beta * spread[t-1] + error
            spread_lag = spread.shift(1).dropna()
            spread_diff = spread.diff().dropna()
            
            # Alinhar séries
            aligned_data = pd.concat([spread_diff, spread_lag], axis=1).dropna()
            y = aligned_data.iloc[:, 0].values
            x = aligned_data.iloc[:, 1].values
              # Regressão
            beta = np.cov(x, y)[0, 1] / np.var(x)
            
            # Half-life
            if beta < 0:
                half_life = -np.log(2) / np.log(1 + beta)
            else:
                half_life = np.inf
                
            return half_life
        except Exception:
            return np.inf
    
    def _analyze_normalizations(self, spread):
        """Analisa diferentes métodos de normalização do spread"""
        # Z-Score padrão
        z_score_standard = (spread - spread.mean()) / spread.std()
        
        # Z-Score robusto (usando mediana e MAD)
        median = spread.median()
        mad = np.median(np.abs(spread - median))
        z_score_robust = (spread - median) / (1.4826 * mad)  # 1.4826 faz MAD equivaler ao std
        
        # Min-Max scaling
        scaler_minmax = MinMaxScaler(feature_range=(-3, 3))
        z_score_minmax = pd.Series(
            scaler_minmax.fit_transform(spread.values.reshape(-1, 1)).flatten(),
            index=spread.index
        )
          # Rolling Z-Score (janela adaptativa)
        window = min(60, len(spread) // 4)
        rolling_mean = spread.rolling(window=window).mean()
        rolling_std = spread.rolling(window=window).std()
        z_score_rolling = (spread - rolling_mean) / rolling_std
        
        # Percentile-based normalization
        z_score_percentile = spread.rank(pct=True).apply(lambda x: stats.norm.ppf(x))
        
        return {
            'z_score_standard': z_score_standard,
            'z_score_robust': z_score_robust,
            'z_score_minmax': z_score_minmax,
            'z_score_rolling': z_score_rolling,
            'z_score_percentile': z_score_percentile,
            'normalization_stats': {
                'standard': {'mean': z_score_standard.mean(), 'std': z_score_standard.std()},
                'robust': {'median': z_score_robust.median(), 'mad': np.median(np.abs(z_score_robust - z_score_robust.median()))},
                'rolling': {'current_mean': rolling_mean.iloc[-1] if not rolling_mean.empty else None}
            }
        }
    
    def _analyze_spread_distributions(self, spread):
        """Analisa distribuições do spread (Normal vs t-Student)"""
        # Remover NaN
        clean_spread = spread.dropna()
        
        if len(clean_spread) < 30:
            return {"error": "Dados insuficientes para análise"}
        
        # Teste de normalidade
        shapiro_stat, shapiro_p = shapiro(clean_spread.values)
        jb_stat, jb_p = jarque_bera(clean_spread.values)
        
        # Fitting Normal
        normal_params = stats.norm.fit(clean_spread.values)
        normal_aic = self._calculate_aic(clean_spread.values, stats.norm, normal_params)
        
        # Fitting t-Student
        t_params = stats.t.fit(clean_spread.values)
        t_aic = self._calculate_aic(clean_spread.values, stats.t, t_params)
        
        # Comparar modelos
        best_distribution = 't_student' if t_aic < normal_aic else 'normal'
        
        # Cálculos de probabilidade para níveis extremos
        threshold_2sigma = 2 * clean_spread.std()
        threshold_3sigma = 3 * clean_spread.std()
        
        if best_distribution == 't_student':
            prob_2sigma = 2 * (1 - stats.t.cdf(threshold_2sigma, *t_params))
            prob_3sigma = 2 * (1 - stats.t.cdf(threshold_3sigma, *t_params))
        else:
            prob_2sigma = 2 * (1 - stats.norm.cdf(threshold_2sigma, *normal_params))
            prob_3sigma = 2 * (1 - stats.norm.cdf(threshold_3sigma, *normal_params))
        
        return {
            'normality_tests': {
                'shapiro': {'statistic': shapiro_stat, 'p_value': shapiro_p},
                'jarque_bera': {'statistic': jb_stat, 'p_value': jb_p},
                'is_normal': jb_p > 0.05
            },
            'distributions': {
                'normal': {'params': normal_params, 'aic': normal_aic},
                't_student': {'params': t_params, 'aic': t_aic}
            },
            'best_distribution': best_distribution,
            'tail_probabilities': {
                '2_sigma': prob_2sigma,
                '3_sigma': prob_3sigma
            },            'descriptive_stats': {
                'skewness': stats.skew(clean_spread.values),
                'kurtosis': stats.kurtosis(clean_spread.values, fisher=True)
            }
        }
    
    def _calculate_aic(self, data, distribution, params):
        """Calcula AIC para seleção de modelo"""
        try:
            log_likelihood = np.sum(distribution.logpdf(data, *params))
            k = len(params)  # número de parâmetros
            aic = 2 * k - 2 * log_likelihood
            return aic
        except Exception:
            return np.inf
    
    def _generate_advanced_signals(self, spread_analysis, distribution_analysis, normalization_analysis):
        """Gera sinais de trading baseados na melhor distribuição"""
        # Usar a melhor normalização (robust é geralmente mais estável)
        normalized_spread = normalization_analysis['z_score_robust']
        
        # Thresholds baseados na distribuição
        best_dist = distribution_analysis['best_distribution']
        
        if best_dist == 't_student':
            # Para t-Student, usar percentis da distribuição
            params = distribution_analysis['distributions']['t_student']['params']
            entry_threshold = abs(stats.t.ppf(0.1, *params))  # 10% nas caudas
            exit_threshold = abs(stats.t.ppf(0.4, *params))   # 40% nas caudas
        else:
            # Para Normal, usar z-scores padrão
            entry_threshold = 2.0
            exit_threshold = 0.5
        
        # Gerar sinais
        signals = pd.Series(0, index=normalized_spread.index)
        
        # Sinais de entrada
        signals[normalized_spread > entry_threshold] = -1  # Short spread (long asset2, short asset1)
        signals[normalized_spread < -entry_threshold] = 1   # Long spread (long asset1, short asset2)
        
        # Sinais de saída
        exit_long = (normalized_spread > -exit_threshold) & (signals.shift(1) == 1)
        exit_short = (normalized_spread < exit_threshold) & (signals.shift(1) == -1)
        
        signals[exit_long | exit_short] = 0
        
        return {
            'signals': signals,
            'normalized_spread': normalized_spread,
            'entry_threshold': entry_threshold,
            'exit_threshold': exit_threshold,
            'best_distribution': best_dist,            'signal_stats': {
                'total_signals': (signals != 0).sum(),
                'long_signals': (signals == 1).sum(),
                'short_signals': (signals == -1).sum()
            }
        }
    
    def create_pair_analysis_plots(self, analysis_result):
        """Cria gráficos para análise de pair trading"""
        if 'error' in analysis_result:
            return None
            
        spread_analysis = analysis_result['spread_analysis']
        distribution_analysis = analysis_result['distribution_analysis']
        trading_signals = analysis_result['trading_signals']
        
        # Criar subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Spread Original', 'Spread Normalizado',
                'Distribuição do Spread', 'Sinais de Trading',
                'Z-Score Rolling', 'Performance Simulada'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Spread original
        spread = spread_analysis['spread']
        fig.add_trace(
            go.Scatter(x=spread.index, y=spread.values, name='Spread', line=dict(color='blue')),
            row=1, col=1
        )
        
        # 2. Spread normalizado
        normalized_spread = trading_signals['normalized_spread']
        entry_threshold = trading_signals['entry_threshold']
        
        fig.add_trace(
            go.Scatter(x=normalized_spread.index, y=normalized_spread.values, 
                      name='Z-Score', line=dict(color='purple')),
            row=1, col=2
        )
        
        # Thresholds
        fig.add_hline(y=entry_threshold, line_dash="dash", line_color="red", 
                     annotation_text="Entry", row=1, col=2)
        fig.add_hline(y=-entry_threshold, line_dash="dash", line_color="red", row=1, col=2)
        
        # 3. Histograma do spread
        fig.add_trace(
            go.Histogram(x=spread.dropna().values, nbinsx=50, name='Spread Hist', 
                        opacity=0.7, histnorm='probability density'),
            row=2, col=1
        )
        
        # 4. Sinais de trading
        signals = trading_signals['signals']
        signal_points = signals[signals != 0]
        
        colors = ['green' if x == 1 else 'red' for x in signal_points.values]
        fig.add_trace(
            go.Scatter(x=signal_points.index, y=normalized_spread.loc[signal_points.index], 
                      mode='markers', name='Trading Signals', 
                      marker=dict(color=colors, size=8)),
            row=2, col=2
        )
        fig.add_trace(
            go.Scatter(x=normalized_spread.index, y=normalized_spread.values, 
                      name='Z-Score', line=dict(color='lightblue', width=1)),
            row=2, col=2
        )
        
        # 5. Z-Score rolling
        rolling_z = spread_analysis.get('rolling_z_score')
        if rolling_z is not None:
            fig.add_trace(
                go.Scatter(x=rolling_z.index, y=rolling_z.values, 
                          name='Rolling Z-Score', line=dict(color='orange')),
                row=3, col=1
            )
        
        # 6. Performance simulada (placeholder)
        cumulative_returns = (signals.shift(1) * normalized_spread * 0.01).cumsum()
        fig.add_trace(
            go.Scatter(x=cumulative_returns.index, y=cumulative_returns.values, 
                      name='Cumulative Returns', line=dict(color='green')),
            row=3, col=2
        )
        
        fig.update_layout(
            height=900,
            title_text="Análise Completa de Pair Trading",
            showlegend=True
        )
        
        return fig
    
    def backtest_pair_strategy(self, analysis_result, transaction_cost=0.001):
        """Executa backtest da estratégia de pair trading"""
        if 'error' in analysis_result:
            return {"error": "Análise inválida"}
            
        signals = analysis_result['trading_signals']['signals']
        normalized_spread = analysis_result['trading_signals']['normalized_spread']
        
        # Simular retornos da estratégia
        position_changes = signals.diff().fillna(0)
        transaction_costs = abs(position_changes) * transaction_cost
        
        # Retornos da estratégia (simplificado)
        strategy_returns = signals.shift(1) * normalized_spread.pct_change() * 0.1
        strategy_returns_net = strategy_returns - transaction_costs
        
        # Métricas de performance
        total_return = strategy_returns_net.sum()
        sharpe_ratio = strategy_returns_net.mean() / strategy_returns_net.std() * np.sqrt(252)
        max_drawdown = (strategy_returns_net.cumsum() - strategy_returns_net.cumsum().expanding().max()).min()
        
        # Análise de trades
        trade_signals = signals[signals != 0]
        num_trades = len(trade_signals)
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,            'num_trades': num_trades,
            'avg_trade_duration': len(signals) / max(num_trades, 1),
            'strategy_returns': strategy_returns_net,
            'cumulative_returns': strategy_returns_net.cumsum()
        }
    
    def test_advanced_cointegration(self, asset1, asset2):
        """
        Realiza teste de cointegração avançado entre dois ativos
        
        Esta função implementa múltiplos testes de cointegração para determinar
        a relação de longo prazo entre dois ativos financeiros, usando
        tanto o método Engle-Granger quanto análise de robustez com janelas temporais
        variáveis. Metodologia baseada nas pesquisas do Prof. Carlos Alberto Rodrigues.
        
        Args:
            asset1 (str): Primeiro ativo do par
            asset2 (str): Segundo ativo do par
            
        Returns:
            dict: Resultados da análise de cointegração
        """
        # Verificar se os ativos existem nos dados
        if asset1 not in self.data.columns or asset2 not in self.data.columns:
            return {
                'is_cointegrated': False,
                'error': f"Um ou mais ativos não encontrados: {asset1}, {asset2}"
            }
            
        # Obter preços alinhados
        prices1 = self.data[asset1].dropna()
        prices2 = self.data[asset2].dropna()
        
        # Alinhar temporalmente
        aligned_data = pd.concat([prices1, prices2], axis=1).dropna()
        if len(aligned_data) < 252:  # Pelo menos um ano de dados
            return {
                'is_cointegrated': False,
                'error': "Dados insuficientes para teste de cointegração"
            }
            
        aligned_prices1 = aligned_data.iloc[:, 0]
        aligned_prices2 = aligned_data.iloc[:, 1]
        
        # Teste Engle-Granger padrão
        try:
            coint_stat, p_value, critical_values = coint(aligned_prices1, aligned_prices2)
            is_cointegrated = p_value < 0.05
        except Exception:
            is_cointegrated = False
            p_value = 1.0
            coint_stat = 0
            critical_values = [0, 0, 0]
            
        # Análise de cointegração em janelas temporais
        window_sizes = [252, 126, 63]  # Aproximadamente 1 ano, 6 meses, 3 meses
        window_results = []
        
        for window in window_sizes:
            if len(aligned_data) >= window:
                try:
                    # Para cada janela, testar a última parte dos dados
                    window_prices1 = aligned_prices1[-window:]
                    window_prices2 = aligned_prices2[-window:]
                    
                    # Teste de cointegração para a janela
                    w_coint_stat, w_p_value, _ = coint(window_prices1, window_prices2)
                    
                    window_results.append({
                        'window_size': window,
                        'is_cointegrated': w_p_value < 0.05,
                        'p_value': w_p_value,
                        'coint_statistic': w_coint_stat
                    })
                except Exception:
                    window_results.append({
                        'window_size': window,
                        'is_cointegrated': False,
                        'error': "Erro no teste de cointegração"
                    })
                    
        # Robustez: teste de phillips-ouliaris
        try:
            from statsmodels.tsa.stattools import kpss
            
            # Regressão OLS para obter resíduos
            x = sm.add_constant(aligned_prices2)
            model = sm.OLS(aligned_prices1, x).fit()
            residuals = model.resid
            
            # Teste KPSS para estacionariedade dos resíduos
            kpss_stat, kpss_p_value, _, _ = kpss(residuals)
            
            # Contrário do usual: p-valor alto = estacionário = cointegrado
            is_kpss_stationary = kpss_p_value > 0.05
            
            phillips_test = {
                'is_stationary': is_kpss_stationary,
                'kpss_statistic': kpss_stat,
                'kpss_p_value': kpss_p_value
            }
        except Exception:
            phillips_test = {
                'error': "Teste de Phillips-Ouliaris falhou"
            }
            
        # Calcular hedge ratio via regressão OLS
        try:
            import statsmodels.api as sm
            x = sm.add_constant(aligned_prices2.values)
            model = sm.OLS(aligned_prices1.values, x).fit()
            hedge_ratio = model.params[1]
            intercept = model.params[0]
            r_squared = model.rsquared
        except Exception:
            # Fallback para método simples
            coeffs = np.polyfit(aligned_prices2.values, aligned_prices1.values, 1)
            hedge_ratio = coeffs[0]
            intercept = coeffs[1]
            residual = aligned_prices1 - (intercept + hedge_ratio * aligned_prices2)
            r_squared = 1 - np.var(residual) / np.var(aligned_prices1)
            
        # Calcular spread e propriedades
        spread = aligned_prices1 - (intercept + hedge_ratio * aligned_prices2)
        spread_mean = spread.mean()
        spread_std = spread.std()
        
        # Avaliação de robustez geral
        windows_cointegrated = sum(1 for w in window_results if w.get('is_cointegrated', False))
        is_robust_cointegrated = is_cointegrated and (windows_cointegrated >= len(window_results) // 2)
        
        return {
            'assets': {'asset1': asset1, 'asset2': asset2},
            'is_cointegrated': is_cointegrated,
            'is_robust_cointegrated': is_robust_cointegrated,
            'p_value': p_value,
            'coint_statistic': coint_stat,
            'critical_values': critical_values,
            'hedge_ratio': hedge_ratio,
            'intercept': intercept,
            'r_squared': r_squared,
            'spread_mean': spread_mean,
            'spread_std': spread_std,
            'window_analysis': window_results,
            'phillips_test': phillips_test
        }
    
    def create_comprehensive_analysis_plot(self, analysis_result):
        """Cria visualização completa da análise"""
        if 'error' in analysis_result:
            return None
        
        # Dados para plotagem
        spread = analysis_result['spread_analysis']['spread']
        z_scores = analysis_result['trading_signals']['z_scores']
        positions = analysis_result['trading_signals']['positions']
        distribution_analysis = analysis_result['distribution_analysis']
        
        # Criar subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Spread vs Hedge Ratio',
                'Z-Score e Sinais de Trading',
                'Distribuição do Spread',
                'Q-Q Plot: Normal vs t-Student',
                'Probabilidades de Eventos Extremos',
                'Análise de Normalizações'
            ],
            specs=[[{"secondary_y": True}, {"secondary_y": True}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # 1. Spread
        fig.add_trace(
            go.Scatter(x=spread.index, y=spread.values, name='Spread', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Média móvel do spread
        spread_ma = spread.rolling(30).mean()
        fig.add_trace(
            go.Scatter(x=spread_ma.index, y=spread_ma.values, name='Spread MA(30)', 
                      line=dict(color='red', dash='dash')),
            row=1, col=1
        )
        
        # 2. Z-Score e sinais
        fig.add_trace(
            go.Scatter(x=z_scores.index, y=z_scores.values, name='Z-Score', line=dict(color='purple')),
            row=1, col=2
        )
        
        # Thresholds
        entry_threshold = analysis_result['trading_signals']['parameters']['entry_threshold']
        fig.add_hline(y=entry_threshold, line_dash="dash", line_color="red", row=1, col=2)
        fig.add_hline(y=-entry_threshold, line_dash="dash", line_color="red", row=1, col=2)
        
        # Sinais de entrada
        entry_signals = positions[positions != 0]
        if len(entry_signals) > 0:
            fig.add_trace(
                go.Scatter(x=entry_signals.index, y=z_scores.loc[entry_signals.index], 
                          mode='markers', name='Trading Signals', 
                          marker=dict(color='red', size=8)),
                row=1, col=2
            )
        
        # 3. Histograma do spread com distribuições ajustadas
        fig.add_trace(
            go.Histogram(x=spread.values, nbinsx=50, histnorm='probability density',
                        name='Spread Distribution', opacity=0.7),
            row=2, col=1
        )
        
        # Distribuições ajustadas
        x_range = np.linspace(spread.min(), spread.max(), 100)
        
        # Normal
        normal_params = distribution_analysis['distribution_fits']['normal']['params']
        normal_pdf = stats.norm.pdf(x_range, *normal_params)
        fig.add_trace(
            go.Scatter(x=x_range, y=normal_pdf, mode='lines', name='Normal Fit',
                      line=dict(color='red', dash='dash')),
            row=2, col=1
        )
        
        # t-Student
        t_params = distribution_analysis['distribution_fits']['t_student']['params']
        t_pdf = stats.t.pdf(x_range, *t_params)
        fig.add_trace(
            go.Scatter(x=x_range, y=t_pdf, mode='lines', name='t-Student Fit',
                      line=dict(color='green', dash='dot')),
            row=2, col=1
        )
        
        # 4. Q-Q Plot
        # Normal Q-Q
        (osm_norm, osr_norm), _ = stats.probplot(spread, dist="norm", plot=None)
        fig.add_trace(
            go.Scatter(x=osm_norm, y=osr_norm, mode='markers', name='Normal Q-Q',
                      marker=dict(color='blue', opacity=0.6)),
            row=2, col=2
        )
        
        # t-Student Q-Q
        df = t_params[0]
        (osm_t, osr_t), _ = stats.probplot(spread, dist=stats.t, sparams=(df,), plot=None)
        fig.add_trace(
            go.Scatter(x=osm_t, y=osr_t, mode='markers', name='t-Student Q-Q',
                      marker=dict(color='red', opacity=0.6)),
            row=2, col=2
        )
        
        # Linha de referência
        fig.add_trace(
            go.Scatter(x=osm_norm, y=osm_norm, mode='lines', name='Perfect Fit',
                      line=dict(color='gray', dash='dash')),
            row=2, col=2
        )
        
        # 5. Probabilidades de eventos extremos
        extreme_probs = distribution_analysis['extreme_probabilities']['probabilities']
        prob_categories = ['2σ Normal', '2σ t-Student', '2σ Empírico', 
                          '3σ Normal', '3σ t-Student', '3σ Empírico']
        prob_values = [
            extreme_probs['normal_2sigma'], extreme_probs['t_student_2sigma'], extreme_probs['empirical_2sigma'],
            extreme_probs['normal_3sigma'], extreme_probs['t_student_3sigma'], extreme_probs['empirical_3sigma']
        ]
        
        fig.add_trace(
            go.Bar(x=prob_categories, y=prob_values, name='Extreme Event Probabilities'),
            row=3, col=1
        )
        
        # 6. Comparação de normalizações
        norm_analysis = analysis_result['normalization_analysis']
        norm_methods = ['Standard', 'Robust', 'MinMax', 'Rolling', 'Percentile']
        norm_stds = [
            norm_analysis['z_score_standard'].std(),
            norm_analysis['z_score_robust'].std(),
            norm_analysis['z_score_minmax'].std(),
            norm_analysis['z_score_rolling'].std(),
            norm_analysis['z_score_percentile'].std()
        ]
        
        fig.add_trace(
            go.Bar(x=norm_methods, y=norm_stds, name='Normalization Stability'),
            row=3, col=2
        )
        
        fig.update_layout(height=1200, showlegend=True, title_text="Análise Completa de Pair Trading")
        
        return fig
    
    def compare_cointegration_methods(self, asset1, asset2):
        """
        Compara diferentes métodos de cointegração para um par de ativos
        
        Esta função implementa e compara três métodos de análise de cointegração:
        1. Teste Engle-Granger (ADF nos resíduos)
        2. Teste de Johansen (para relações multivariadas)
        3. Método Phillips-Ouliaris 
        
        Args:
            asset1 (str): Primeiro ativo do par
            asset2 (str): Segundo ativo do par
            
        Returns:
            dict: Resultados comparativos dos diferentes métodos
        """
        from statsmodels.tsa.vector_ar.vecm import coint_johansen
        from statsmodels.regression.linear_model import OLS
        import statsmodels.api as sm
        
        # Verificar se os ativos existem nos dados
        if asset1 not in self.data.columns or asset2 not in self.data.columns:
            return {
                'is_cointegrated': False,
                'error': f"Um ou mais ativos não encontrados: {asset1}, {asset2}"
            }
        
        # Obter preços
        price1 = self.data[asset1].dropna()
        price2 = self.data[asset2].dropna()
        
        # Alinhar temporalmente
        aligned = pd.concat([price1, price2], axis=1).dropna()
        if len(aligned) < 252:  # Pelo menos um ano de dados
            return {
                'is_cointegrated': False,
                'error': "Dados insuficientes para teste de cointegração"
            }
        
        results = {}
        
        # 1. Método Engle-Granger
        try:
            # Teste padrão Engle-Granger
            coint_t_stat, p_value, critical_values = coint(aligned[asset1], aligned[asset2])
            
            # Regressão OLS para calcular o hedge ratio
            model = OLS(aligned[asset1], sm.add_constant(aligned[asset2])).fit()
            resid = model.resid
            
            # Teste ADF nos resíduos
            adf_result = adfuller(resid)
            
            eg_result = {
                'method': 'Engle-Granger',
                'is_cointegrated': p_value < 0.05,
                'p_value': p_value,
                't_statistic': coint_t_stat,
                'critical_values': critical_values,
                'hedge_ratio': model.params[1],
                'intercept': model.params[0],
                'adf_statistic': adf_result[0],
                'adf_p_value': adf_result[1]
            }
            results['engle_granger'] = eg_result
        except Exception as e:
            results['engle_granger'] = {'error': str(e)}
            
        # 2. Método Johansen
        try:
            # Preparar dados para teste Johansen
            data_matrix = aligned.values
            
            # Aplicar teste Johansen
            johansen_result = coint_johansen(data_matrix, det_order=0, k_ar_diff=1)
            
            # Verificar resultados utilizando estatísticas de traço
            trace_stat = johansen_result.lr1[0]  # Estatística de traço para r=0
            critical_values_90 = johansen_result.cvt[0, 0]  # 90% valor crítico
            critical_values_95 = johansen_result.cvt[0, 1]  # 95% valor crítico
            critical_values_99 = johansen_result.cvt[0, 2]  # 99% valor crítico
            
            # Verificar cointegração (estatística de traço > valor crítico)
            is_coint_90 = trace_stat > critical_values_90
            is_coint_95 = trace_stat > critical_values_95
            is_coint_99 = trace_stat > critical_values_99
            
            # Extrair vetor de cointegração (se houver)
            if is_coint_95:
                coint_vector = johansen_result.evec[:, 0]
                # Normalizar para interpretar como hedge ratio
                hedge_ratio = -coint_vector[1] / coint_vector[0]
            else:
                hedge_ratio = None
                
            johansen_result_dict = {
                'method': 'Johansen',
                'is_cointegrated': is_coint_95,
                'trace_statistic': trace_stat,
                'critical_values': {
                    '90%': critical_values_90,
                    '95%': critical_values_95,
                    '99%': critical_values_99
                },
                'hedge_ratio': hedge_ratio,
                'cointegrating_vector': None if not is_coint_95 else coint_vector.tolist()
            }
            results['johansen'] = johansen_result_dict
        except Exception as e:
            results['johansen'] = {'error': str(e)}
            
        # 3. Método Phillips-Ouliaris (aproximado via statsmodels)
        try:
            from statsmodels.tsa.stattools import kpss
            
            # Regressão OLS
            model_po = sm.OLS(aligned[asset1], sm.add_constant(aligned[asset2])).fit()
            resid_po = model_po.resid
            
            # Phillips-Perron teste nos resíduos 
            kpss_result = kpss(resid_po)
            
            # Contrário do usual: p-valor alto = estacionário = cointegrado
            is_kpss_stationary = kpss_result[1] > 0.05
            
            phillips_result = {
                'method': 'Phillips-Ouliaris (aproximado)',
                'is_stationary': is_kpss_stationary,
                'kpss_statistic': kpss_result[0],
                'kpss_p_value': kpss_result[1],
                'hedge_ratio': model_po.params[1],
                'intercept': model_po.params[0],
                'residual_autocorrelation': pd.Series(resid_po).autocorr()
            }
            results['phillips_ouliaris'] = phillips_result
        except Exception as e:
            results['phillips_ouliaris'] = {'error': str(e)}
            
        # 4. Sumário comparativo
        methods_agreement = sum(1 for method in results.values() 
                              if isinstance(method, dict) and method.get('is_cointegrated', False))
        
        is_robust_cointegrated = methods_agreement >= 2  # Pelo menos dois métodos concordam
        
        summary = {
            'is_robust_cointegrated': is_robust_cointegrated,
            'methods_agreement_count': methods_agreement,
            'total_methods': len(results),
            'best_method': max(results.keys(), 
                              key=lambda k: 0 if not isinstance(results[k], dict) or 'error' in results[k] 
                              else (1 if results[k].get('is_cointegrated', False) else 0)),
            'recommendation': "Usar o par" if is_robust_cointegrated else "Evitar o par"
        }
        
        return {
            'assets': {'asset1': asset1, 'asset2': asset2},
            'methods_results': results,
            'summary': summary
        }
