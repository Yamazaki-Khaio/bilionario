#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESTRATÉGIA DE PAIR TRADING - ANÁLISE COMPLETA
==============================================

SUMÁRIO DE NAVEGAÇÃO:
1. [IMPORTS E CONFIGURAÇÕES] - Importações necessárias e configurações iniciais
2. [COLETA DE DADOS] - Download de dados históricos (5+ anos)
3. [ANÁLISE DE CORRELAÇÃO] - Identificação dos pares mais correlacionados
4. [TESTE DE COINTEGRAÇÃO] - Validação estatística dos pares
5. [ESTRATÉGIA DE TRADING] - Geração de sinais baseada em Z-Score
6. [BACKTEST E MÉTRICAS] - Simulação da estratégia com métricas de performance
7. [VISUALIZAÇÃO] - Gráficos de análise e resultados
8. [ANÁLISE MÚLTIPLOS PARES] - Comparação entre diferentes pares
9. [OTIMIZAÇÃO DE PARÂMETROS] - Fine-tuning dos thresholds

Autor: Kaio Geovan
Data: Junho 2025
"""

# ==============================================================================
# 1. [IMPORTS E CONFIGURAÇÕES]
# ==============================================================================

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.vector_ar.vecm import coint_johansen
import warnings
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from financial_formatting import format_percentage, format_ratio

# Configurações
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
np.random.seed(42)

# ==============================================================================
# 2. [COLETA DE DADOS]
# ==============================================================================

class PairTradingStrategy:
    """
    Classe principal para implementação da estratégia de Pair Trading
    
    A estratégia de Pair Trading é baseada na teoria de reversão à média,
    onde buscamos ativos cointegrados que tendem a convergir no longo prazo.
    """
    
    def __init__(self, start_date='2019-01-01'):
        """
        Inicializa a estratégia com período de 5+ anos
        
        Args:
            start_date (str): Data inicial para coleta de dados
        """
        self.start_date = start_date
        self.end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Ativos incluindo forex, cripto e brasileiros
        self.tickers = {
            'forex': ['USDBRL=X', 'EURBRL=X', 'GBPBRL=X', 'JPYBRL=X'],
            'cripto': ['BTC-USD', 'ETH-USD', 'ADA-USD', 'SOL-USD', 'XRP-USD'],
            'acoes_br': ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA', 'ABEV3.SA',
                        'BBAS3.SA', 'WEGE3.SA', 'RENT3.SA', 'RADL3.SA', 'LREN3.SA'],
            'etfs': ['BOVA11.SA', 'SMAL11.SA', 'IVVB11.SA']
        }
        
        # Combinar todos os tickers
        self.all_tickers = []
        for category in self.tickers.values():
            self.all_tickers.extend(category)
            
        self.data = None
        self.returns = None
        self.correlation_matrix = None
        self.best_pairs = []
        
    def fetch_data(self):
        """
        Coleta dados históricos dos ativos
        
        Justificativa: 
        - Período de 5+ anos garante robustez estatística
        - Preços ajustados eliminam distorções de dividendos/splits
        - Tratamento de dados faltantes por forward fill
        """
        print(f"📊 Coletando dados de {self.start_date} até {self.end_date}")
        print(f"📈 Total de ativos: {len(self.all_tickers)}")
        
        try:
            # Download dos dados com preços ajustados
            self.data = yf.download(
                self.all_tickers, 
                start=self.start_date, 
                end=self.end_date,
                auto_adjust=True,
                progress=True
            )['Close']
            
            # Tratamento de dados faltantes
            self.data = self.data.fillna(method='ffill').dropna()
            
            # Calcular retornos logarítmicos (mais apropriados para análise financeira)
            self.returns = np.log(self.data / self.data.shift(1)).dropna()
            
            print(f"✅ Dados coletados: {len(self.data)} observações")
            print(f"📅 Período: {self.data.index[0].date()} até {self.data.index[-1].date()}")
            
            return True
            
        except Exception as e:
            print(f"❌ Erro ao coletar dados: {e}")
            return False

# ==============================================================================
# 3. [ANÁLISE DE CORRELAÇÃO]
# ==============================================================================

    def analyze_correlations(self, min_correlation=0.7):
        """
        Analisa correlações entre ativos para identificar candidatos a pair trading
        
        Justificativa Estatística:
        - Correlação > 0.7 indica forte relação linear
        - Análise de correlação rolling identifica estabilidade temporal
        - Excluímos auto-correlações (diagonal = 1)
        
        Args:
            min_correlation (float): Correlação mínima para considerar um par
        """
        print(f"\n🔍 Analisando correlações (mínimo: {min_correlation})")
        
        # Matriz de correlação
        self.correlation_matrix = self.returns.corr()
        
        # Identificar pares com alta correlação
        pairs = []
        n_assets = len(self.correlation_matrix.columns)
        
        for i in range(n_assets):
            for j in range(i+1, n_assets):
                asset1 = self.correlation_matrix.columns[i]
                asset2 = self.correlation_matrix.columns[j]
                corr = self.correlation_matrix.iloc[i, j]
                
                if abs(corr) >= min_correlation:
                    pairs.append({
                        'asset1': asset1,
                        'asset2': asset2,
                        'correlation': corr,
                        'abs_correlation': abs(corr)
                    })
        
        # Ordenar por correlação absoluta
        pairs = sorted(pairs, key=lambda x: x['abs_correlation'], reverse=True)
        self.candidate_pairs = pairs[:10]  # Top 10 pares
        
        print(f"📊 Pares identificados com correlação ≥ {min_correlation}: {len(self.candidate_pairs)}")
        
        # Exibir top 5
        print("\n🏆 TOP 5 PARES MAIS CORRELACIONADOS:")
        for i, pair in enumerate(self.candidate_pairs[:5]):
            print(f"{i+1}. {pair['asset1']} vs {pair['asset2']}: {pair['correlation']:.3f}")
            
        return self.candidate_pairs

# ==============================================================================
# 4. [TESTE DE COINTEGRAÇÃO]
# ==============================================================================

    def test_cointegration(self, asset1, asset2, critical_level=0.05):
        """
        Teste de cointegração usando método de Engle-Granger
        
        Justificativa Estatística:
        - Cointegração indica relação de equilíbrio de longo prazo
        - Teste de Engle-Granger: se resíduos da regressão são estacionários,
          as séries são cointegradas
        - p-valor < 0.05 rejeita hipótese nula (não cointegração)
        
        Args:
            asset1, asset2 (str): Símbolos dos ativos
            critical_level (float): Nível crítico para teste (α = 0.05)
            
        Returns:
            dict: Resultados do teste de cointegração
        """
        price1 = self.data[asset1].dropna()
        price2 = self.data[asset2].dropna()
        
        # Alinhar séries temporais
        common_index = price1.index.intersection(price2.index)
        price1 = price1[common_index]
        price2 = price2[common_index]
        
        # Teste de cointegração de Engle-Granger
        coint_stat, p_value, critical_values = coint(price1, price2)
        
        # Calcular hedge ratio via regressão OLS
        # Justificativa: minimiza variância do spread
        model = OLS(price1, price2).fit()
        hedge_ratio = model.params[0]
        
        # Calcular spread
        spread = price1 - hedge_ratio * price2
        spread_mean = spread.mean()
        spread_std = spread.std()
        
        # Teste de estacionariedade do spread (ADF)
        adf_stat, adf_p_value, _, _, adf_critical, _ = adfuller(spread)
        
        result = {
            'asset1': asset1,
            'asset2': asset2,
            'price1': price1,
            'price2': price2,
            'hedge_ratio': hedge_ratio,
            'spread': spread,
            'spread_mean': spread_mean,
            'spread_std': spread_std,
            'coint_stat': coint_stat,
            'p_value': p_value,
            'critical_values': critical_values,
            'is_cointegrated': p_value < critical_level,
            'adf_stat': adf_stat,
            'adf_p_value': adf_p_value,
            'adf_critical': adf_critical,
            'r_squared': model.rsquared
        }
        
        return result

# ==============================================================================
# 5. [ESTRATÉGIA DE TRADING]
# ==============================================================================

    def generate_trading_signals(self, coint_result, entry_threshold=2.0, exit_threshold=0.5):
        """
        Gera sinais de trading baseados no Z-Score do spread
        
        Justificativa Estatística:
        - Z-Score normaliza o spread pela volatilidade histórica
        - Thresholds baseados em desvios padrão (2σ = 95% dos dados)
        - Estratégia de reversão à média: comprar spread "barato", vender "caro"
        
        Lógica dos Sinais:
        - Z-Score > +2σ: Spread alto → Short Asset1, Long Asset2
        - Z-Score < -2σ: Spread baixo → Long Asset1, Short Asset2  
        - |Z-Score| < 0.5σ: Sair da posição
        
        Args:
            coint_result (dict): Resultado do teste de cointegração
            entry_threshold (float): Threshold de entrada (em desvios padrão)
            exit_threshold (float): Threshold de saída
            
        Returns:
            pd.DataFrame: Sinais de trading
        """
        spread = coint_result['spread']
        spread_mean = coint_result['spread_mean']
        spread_std = coint_result['spread_std']
        
        # Calcular Z-Score (normalização estatística)
        z_score = (spread - spread_mean) / spread_std
        
        # DataFrame para sinais
        signals = pd.DataFrame(index=spread.index)
        signals['spread'] = spread
        signals['z_score'] = z_score
        signals['signal'] = 0  # 0: sem posição, 1: long spread, -1: short spread
        signals['position'] = 0
        
        # Geração de sinais baseada em thresholds
        current_position = 0
        
        for i in range(len(signals)):
            z = z_score.iloc[i]
            
            # Lógica de entrada
            if current_position == 0:
                if z > entry_threshold:
                    current_position = -1  # Short spread (sell asset1, buy asset2)
                elif z < -entry_threshold:
                    current_position = 1   # Long spread (buy asset1, sell asset2)
            
            # Lógica de saída
            elif abs(z) < exit_threshold:
                current_position = 0
            
            signals.iloc[i, signals.columns.get_loc('position')] = current_position
        
        # Identificar pontos de entrada e saída
        signals['signal'] = signals['position'].diff()
        
        return signals

# ==============================================================================
# 6. [BACKTEST E MÉTRICAS]
# ==============================================================================

    def backtest_strategy(self, coint_result, signals, initial_capital=100000, 
                         transaction_cost=0.001):
        """
        Executa backtest da estratégia com cálculo de métricas de performance
        
        Justificativa das Métricas:
        - Sharpe Ratio: retorno ajustado ao risco (>1 é bom, >2 é excelente)
        - Maximum Drawdown: maior perda de pico a vale (risco de ruína)
        - Win Rate: % de trades lucrativos
        - Profit Factor: razão lucros/perdas
        
        Args:
            coint_result (dict): Resultado do teste de cointegração
            signals (pd.DataFrame): Sinais de trading
            initial_capital (float): Capital inicial
            transaction_cost (float): Custo de transação por trade
            
        Returns:
            dict: Métricas de performance
        """
        price1 = coint_result['price1']
        price2 = coint_result['price2']
        hedge_ratio = coint_result['hedge_ratio']
        
        # Calcular retornos dos ativos
        returns1 = price1.pct_change().fillna(0)
        returns2 = price2.pct_change().fillna(0)
        
        # Retornos da estratégia
        # Posição 1: Long spread = Long asset1, Short asset2
        # Posição -1: Short spread = Short asset1, Long asset2
        strategy_returns = []
        
        for i in range(len(signals)):
            position = signals['position'].iloc[i]
            
            if position != 0:
                # Retorno do spread = ret1 - hedge_ratio * ret2
                if position == 1:  # Long spread
                    ret = returns1.iloc[i] - hedge_ratio * returns2.iloc[i]
                else:  # Short spread
                    ret = -(returns1.iloc[i] - hedge_ratio * returns2.iloc[i])
                
                # Aplicar custo de transação nas mudanças de posição
                if i > 0 and signals['position'].iloc[i] != signals['position'].iloc[i-1]:
                    ret -= transaction_cost
                
                strategy_returns.append(ret)
            else:
                strategy_returns.append(0)
        
        strategy_returns = pd.Series(strategy_returns, index=signals.index)
        
        # Curva de equity
        equity_curve = (1 + strategy_returns).cumprod() * initial_capital
        
        # Métricas de performance
        total_return = (equity_curve.iloc[-1] / initial_capital) - 1
        
        # Retorno e volatilidade anualizados
        trading_days = 252
        n_years = len(strategy_returns) / trading_days
        annual_return = (1 + total_return) ** (1/n_years) - 1
        annual_volatility = strategy_returns.std() * np.sqrt(trading_days)
        
        # Sharpe Ratio (assumindo risk-free rate = 0)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
        
        # Maximum Drawdown
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Análise de trades
        trades = strategy_returns[strategy_returns != 0]
        num_trades = len(trades)
        winning_trades = trades[trades > 0]
        losing_trades = trades[trades < 0]
        
        win_rate = len(winning_trades) / num_trades if num_trades > 0 else 0
        avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades.mean() if len(losing_trades) > 0 else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # Número de mudanças de posição
        position_changes = (signals['position'].diff() != 0).sum()
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'equity_curve': equity_curve,
            'strategy_returns': strategy_returns,
            'drawdown_series': drawdown,
            'num_trades': num_trades,
            'position_changes': position_changes,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'trades': trades
        }

# ==============================================================================
# 7. [VISUALIZAÇÃO]
# ==============================================================================

    def plot_comprehensive_analysis(self, coint_result, signals, backtest_result):
        """
        Cria visualização completa da análise de pair trading
        """
        fig = make_subplots(
            rows=4, cols=2,
            subplot_titles=[
                'Preços Normalizados dos Ativos', 'Correlação Rolling (60 dias)',
                'Spread e Z-Score', 'Sinais de Trading',
                'Equity Curve', 'Drawdown',
                'Distribuição dos Retornos', 'Análise Risk-Return'
            ],
            specs=[[{}, {}],
                   [{"secondary_y": True}, {}],
                   [{}, {}],
                   [{}, {}]],
            vertical_spacing=0.08
        )
        
        # 1. Preços normalizados
        price1_norm = coint_result['price1'] / coint_result['price1'].iloc[0]
        price2_norm = coint_result['price2'] / coint_result['price2'].iloc[0]
        
        fig.add_trace(
            go.Scatter(x=price1_norm.index, y=price1_norm,
                      name=coint_result['asset1'], line=dict(color='blue')),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=price2_norm.index, y=price2_norm,
                      name=coint_result['asset2'], line=dict(color='red')),
            row=1, col=1
        )
        
        # 2. Correlação rolling
        rolling_corr = self.returns[coint_result['asset1']].rolling(60).corr(
            self.returns[coint_result['asset2']]
        )
        fig.add_trace(
            go.Scatter(x=rolling_corr.index, y=rolling_corr,
                      name='Correlação 60d', line=dict(color='green')),
            row=1, col=2
        )
        
        # 3. Spread e Z-Score
        fig.add_trace(
            go.Scatter(x=signals.index, y=signals['spread'],
                      name='Spread', line=dict(color='orange')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=signals.index, y=signals['z_score'],
                      name='Z-Score', line=dict(color='purple')),
            row=2, col=1, secondary_y=True
        )
        
        # Linhas de threshold
        fig.add_hline(y=2, line_dash="dash", line_color="red", row=2, col=1, secondary_y=True)
        fig.add_hline(y=-2, line_dash="dash", line_color="red", row=2, col=1, secondary_y=True)
        fig.add_hline(y=0, line_dash="dot", line_color="black", row=2, col=1, secondary_y=True)
        
        # 4. Sinais de trading
        buy_signals = signals[signals['signal'] == 1]
        sell_signals = signals[signals['signal'] == -1]
        
        fig.add_trace(
            go.Scatter(x=signals.index, y=signals['z_score'],
                      name='Z-Score', line=dict(color='gray')),
            row=2, col=2
        )
        
        if len(buy_signals) > 0:
            fig.add_trace(
                go.Scatter(x=buy_signals.index, y=buy_signals['z_score'],
                          mode='markers', name='Buy Signal',
                          marker=dict(color='green', size=10, symbol='triangle-up')),
                row=2, col=2
            )
        
        if len(sell_signals) > 0:
            fig.add_trace(
                go.Scatter(x=sell_signals.index, y=sell_signals['z_score'],
                          mode='markers', name='Sell Signal',
                          marker=dict(color='red', size=10, symbol='triangle-down')),
                row=2, col=2
            )
        
        # 5. Equity Curve
        fig.add_trace(
            go.Scatter(x=backtest_result['equity_curve'].index,
                      y=backtest_result['equity_curve'],
                      name='Equity Curve', line=dict(color='darkgreen')),
            row=3, col=1
        )
        
        # 6. Drawdown
        fig.add_trace(
            go.Scatter(x=backtest_result['drawdown_series'].index,
                      y=backtest_result['drawdown_series'] * 100,
                      name='Drawdown %', fill='tonexty', 
                      line=dict(color='red')),
            row=3, col=2
        )
        
        # 7. Distribuição dos retornos
        returns_data = backtest_result['strategy_returns'][
            backtest_result['strategy_returns'] != 0
        ] * 100
        
        fig.add_trace(
            go.Histogram(x=returns_data, name='Retornos (%)',
                        nbinsx=30, marker_color='lightblue'),
            row=4, col=1
        )
        
        # 8. Risk-Return scatter
        ann_ret = backtest_result['annual_return'] * 100
        ann_vol = backtest_result['annual_volatility'] * 100
        
        fig.add_trace(
            go.Scatter(x=[ann_vol], y=[ann_ret],
                      mode='markers+text',
                      text=[f"Sharpe: {backtest_result['sharpe_ratio']:.2f}"],
                      textposition='top center',
                      marker=dict(size=15, color='gold'),
                      name='Estratégia'),
            row=4, col=2
        )
        
        # Layout
        fig.update_layout(
            title=f"📊 Análise Completa: {coint_result['asset1']} vs {coint_result['asset2']}",
            height=1200,
            showlegend=True
        )
        
        return fig

# ==============================================================================
# 8. [ANÁLISE MÚLTIPLOS PARES]
# ==============================================================================

    def analyze_multiple_pairs(self, top_n=5):
        """
        Analisa múltiplos pares para encontrar a melhor estratégia
        """
        results = []
        
        print(f"\n🔍 Analisando {top_n} melhores pares...")
        
        for i, pair_info in enumerate(self.candidate_pairs[:top_n]):
            asset1 = pair_info['asset1']
            asset2 = pair_info['asset2']
            
            print(f"\n{i+1}. Analisando par: {asset1} vs {asset2}")
            
            # Teste de cointegração
            coint_result = self.test_cointegration(asset1, asset2)
            
            if coint_result['is_cointegrated']:
                print(f"   ✅ Cointegrado (p-valor: {coint_result['p_value']:.4f})")
                
                # Gerar sinais e fazer backtest
                signals = self.generate_trading_signals(coint_result)
                backtest = self.backtest_strategy(coint_result, signals)
                
                # Compilar resultados
                result = {
                    'asset1': asset1,
                    'asset2': asset2,
                    'correlation': pair_info['correlation'],
                    'cointegration_p_value': coint_result['p_value'],
                    'r_squared': coint_result['r_squared'],
                    'total_return': backtest['total_return'],
                    'annual_return': backtest['annual_return'],
                    'sharpe_ratio': backtest['sharpe_ratio'],
                    'max_drawdown': backtest['max_drawdown'],
                    'win_rate': backtest['win_rate'],
                    'num_trades': backtest['num_trades'],
                    'profit_factor': backtest['profit_factor'],
                    'coint_result': coint_result,
                    'signals': signals,
                    'backtest': backtest
                }
                
                results.append(result)
                
                print(f"   📈 Retorno Total: {backtest['total_return']:.2%}")
                print(f"   ⚡ Sharpe Ratio: {backtest['sharpe_ratio']:.2f}")
                print(f"   📉 Max Drawdown: {backtest['max_drawdown']:.2%}")
                
            else:
                print(f"   ❌ Não cointegrado (p-valor: {coint_result['p_value']:.4f})")
        
        # Ordenar por Sharpe Ratio
        results = sorted(results, key=lambda x: x['sharpe_ratio'], reverse=True)
        
        return results

# ==============================================================================
# 9. [OTIMIZAÇÃO DE PARÂMETROS]
# ==============================================================================

    def optimize_parameters(self, coint_result, entry_range=(1.5, 3.0), 
                           exit_range=(0.3, 1.0), step=0.1):
        """
        Otimiza parâmetros de entrada e saída para maximizar Sharpe Ratio
        """
        print("\n🎯 Otimizando parâmetros...")
        
        best_sharpe = -np.inf
        best_params = {}
        optimization_results = []
        
        entry_values = np.arange(entry_range[0], entry_range[1] + step, step)
        exit_values = np.arange(exit_range[0], exit_range[1] + step, step)
        
        for entry_thresh in entry_values:
            for exit_thresh in exit_values:
                if exit_thresh >= entry_thresh:
                    continue
                    
                signals = self.generate_trading_signals(
                    coint_result, entry_thresh, exit_thresh
                )
                backtest = self.backtest_strategy(coint_result, signals)
                
                optimization_results.append({
                    'entry_threshold': entry_thresh,
                    'exit_threshold': exit_thresh,
                    'sharpe_ratio': backtest['sharpe_ratio'],
                    'total_return': backtest['total_return'],
                    'max_drawdown': backtest['max_drawdown'],
                    'num_trades': backtest['num_trades']
                })
                
                if backtest['sharpe_ratio'] > best_sharpe:
                    best_sharpe = backtest['sharpe_ratio']
                    best_params = {
                        'entry_threshold': entry_thresh,
                        'exit_threshold': exit_thresh,
                        'signals': signals,
                        'backtest': backtest
                    }
        
        print(f"🏆 Melhores parâmetros:")
        print(f"   Entry Threshold: {best_params['entry_threshold']:.1f}")
        print(f"   Exit Threshold: {best_params['exit_threshold']:.1f}")
        print(f"   Sharpe Ratio: {best_sharpe:.2f}")
        
        return best_params, optimization_results

# ==============================================================================
# EXECUÇÃO PRINCIPAL
# ==============================================================================

def main():
    """
    Função principal que executa toda a análise de pair trading
    """
    print("=" * 80)
    print("🚀 ESTRATÉGIA DE PAIR TRADING - ANÁLISE COMPLETA")
    print("=" * 80)
    
    # Inicializar estratégia
    strategy = PairTradingStrategy(start_date='2019-01-01')
    
    # 1. Coletar dados
    if not strategy.fetch_data():
        return
    
    # 2. Analisar correlações
    candidate_pairs = strategy.analyze_correlations(min_correlation=0.7)
    
    if not candidate_pairs:
        print("❌ Nenhum par com correlação suficiente encontrado.")
        return
    
    # 3. Analisar múltiplos pares
    results = strategy.analyze_multiple_pairs(top_n=5)
    
    if not results:
        print("❌ Nenhum par cointegrado encontrado.")
        return
    
    # 4. Selecionar melhor par
    best_pair = results[0]
    print(f"\n🏆 MELHOR PAR: {best_pair['asset1']} vs {best_pair['asset2']}")
    print(f"📊 Sharpe Ratio: {best_pair['sharpe_ratio']:.2f}")
    print(f"📈 Retorno Anual: {best_pair['annual_return']:.2%}")
    print(f"📉 Max Drawdown: {best_pair['max_drawdown']:.2%}")
    print(f"🎯 Win Rate: {best_pair['win_rate']:.2%}")
    
    # 5. Otimizar parâmetros
    optimized_params, opt_results = strategy.optimize_parameters(
        best_pair['coint_result']
    )
    
    # 6. Resultados finais com parâmetros otimizados
    final_signals = optimized_params['signals']
    final_backtest = optimized_params['backtest']
    print(f"\n📊 RESULTADOS FINAIS (OTIMIZADOS):")
    print(f"💰 Capital Inicial: R$ 100.000")
    print(f"💵 Capital Final: R$ {final_backtest['equity_curve'].iloc[-1]:,.2f}")
    print(f"📈 Retorno Total: {final_backtest['total_return']:.2%}")
    print(f"📅 Retorno Anualizado: {final_backtest['annual_return']:.2%}")
    
    # Calcular retorno mensal baseado no retorno anualizado
    monthly_return = (1 + final_backtest['annual_return']) ** (1/12) - 1
    print(f"📊 Retorno Mensal: {monthly_return:.2%}")
    
    print(f"⚡ Sharpe Ratio: {final_backtest['sharpe_ratio']:.2f}")
    print(f"📉 Max Drawdown: {final_backtest['max_drawdown']:.2%}")
    print(f"🔢 Número de Trades: {final_backtest['num_trades']}")
    print(f"🎯 Win Rate: {final_backtest['win_rate']:.2%}")
    print(f"💡 Profit Factor: {final_backtest['profit_factor']:.2f}")
    
    # 7. Visualização
    print(f"\n📊 Gerando visualizações...")
    
    # Gráfico completo
    fig = strategy.plot_comprehensive_analysis(
        best_pair['coint_result'], 
        final_signals, 
        final_backtest
    )
    fig.show()
    
    # Resumo dos resultados
    summary_df = pd.DataFrame(results)
    print(f"\n📋 RESUMO COMPARATIVO DOS PARES:")
    print(summary_df[['asset1', 'asset2', 'sharpe_ratio', 'total_return', 
                     'max_drawdown', 'win_rate']].round(3))
    
    return strategy, results, optimized_params

if __name__ == "__main__":
    # Executar análise
    strategy, results, optimized_params = main()
    
    print(f"\n✅ Análise concluída com sucesso!")
    print(f"📊 Utilize os objetos 'strategy', 'results' e 'optimized_params' para análises adicionais.")