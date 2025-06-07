import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from scipy import stats
from statsmodels.tsa.stattools import coint
import warnings
warnings.filterwarnings('ignore')

class PairTradingAnalysis:
    """Análise e estratégia de Pair Trading"""
    
    def __init__(self, price_data):
        """
        Inicializa análise de Pair Trading
        
        Args:
            price_data (pd.DataFrame): DataFrame com preços dos ativos
        """
        self.price_data = price_data
        self.returns_data = price_data.pct_change().dropna()
        self.correlation_matrix = None
        self.cointegration_results = {}
        self.selected_pairs = []
        
    def find_correlated_pairs(self, min_correlation=0.7, min_years=5):
        """
        Encontra pares de ativos correlacionados por período mínimo
        
        Args:
            min_correlation (float): Correlação mínima entre os ativos
            min_years (int): Período mínimo de dados em anos
            
        Returns:
            list: Lista de pares correlacionados
        """
        # Verificar se temos dados suficientes
        min_periods = min_years * 252  # aproximadamente 252 dias úteis por ano
        if len(self.price_data) < min_periods:
            st.warning(f"Dados insuficientes. Necessário pelo menos {min_years} anos de dados.")
            return []
        
        # Calcular matriz de correlação
        self.correlation_matrix = self.returns_data.corr()
        
        # Encontrar pares correlacionados
        pairs = []
        assets = self.correlation_matrix.columns
        
        for i in range(len(assets)):
            for j in range(i+1, len(assets)):
                asset1, asset2 = assets[i], assets[j]
                correlation = self.correlation_matrix.loc[asset1, asset2]
                
                if abs(correlation) >= min_correlation:
                    pairs.append({
                        'asset1': asset1,
                        'asset2': asset2,
                        'correlation': correlation,
                        'abs_correlation': abs(correlation)
                    })
        
        # Ordenar por correlação absoluta
        pairs = sorted(pairs, key=lambda x: x['abs_correlation'], reverse=True)
        self.selected_pairs = pairs
        
        return pairs
    
    def test_cointegration(self, asset1, asset2):
        """
        Testa cointegração entre dois ativos usando teste de Engle-Granger
        
        Args:
            asset1 (str): Nome do primeiro ativo
            asset2 (str): Nome do segundo ativo
            
        Returns:
            dict: Resultados do teste de cointegração
        """
        try:
            price1 = self.price_data[asset1].dropna()
            price2 = self.price_data[asset2].dropna()
            
            # Alinhar séries temporais
            common_index = price1.index.intersection(price2.index)
            price1 = price1[common_index]
            price2 = price2[common_index]
            
            # Teste de cointegração
            coint_stat, p_value, critical_values = coint(price1, price2)
            
            # Regressão linear para calcular spread
            slope, intercept, r_value, _, _ = stats.linregress(price1, price2)
            spread = price2 - (slope * price1 + intercept)
            
            # Estatísticas do spread
            spread_mean = spread.mean()
            spread_std = spread.std()
            
            result = {
                'asset1': asset1,
                'asset2': asset2,
                'cointegration_stat': coint_stat,
                'p_value': p_value,
                'critical_values': critical_values,
                'is_cointegrated': p_value < 0.05,
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value**2,
                'spread': spread,
                'spread_mean': spread_mean,
                'spread_std': spread_std,
                'price1': price1,
                'price2': price2
            }
            
            self.cointegration_results[f"{asset1}_{asset2}"] = result
            return result
            
        except Exception as e:
            st.error(f"Erro no teste de cointegração para {asset1} vs {asset2}: {str(e)}")
            return None
    
    def generate_trading_signals(self, coint_result, entry_threshold=2.0, exit_threshold=0.5):
        """
        Gera sinais de trading baseados no spread
        
        Args:
            coint_result (dict): Resultado do teste de cointegração
            entry_threshold (float): Limiar para entrada (em desvios padrão)
            exit_threshold (float): Limiar para saída (em desvios padrão)
            
        Returns:
            pd.DataFrame: DataFrame com sinais de trading
        """
        spread = coint_result['spread']
        spread_mean = coint_result['spread_mean']
        spread_std = coint_result['spread_std']
        
        # Normalizar spread (z-score)
        z_score = (spread - spread_mean) / spread_std
        
        # Gerar sinais
        signals = pd.DataFrame(index=spread.index)
        signals['spread'] = spread
        signals['z_score'] = z_score
        signals['signal'] = 0
        signals['position'] = 0
        
        # Lógica de entrada e saída
        for i in range(1, len(signals)):
            # Entrada: spread muito alto (vender asset2, comprar asset1)
            if z_score.iloc[i] > entry_threshold:
                signals['signal'].iloc[i] = -1
            # Entrada: spread muito baixo (comprar asset2, vender asset1)
            elif z_score.iloc[i] < -entry_threshold:
                signals['signal'].iloc[i] = 1
            # Saída: spread voltando ao normal
            elif abs(z_score.iloc[i]) < exit_threshold:
                signals['signal'].iloc[i] = 0
            else:
                signals['signal'].iloc[i] = signals['signal'].iloc[i-1]
        
        # Calcular posições
        signals['position'] = signals['signal'].shift(1).fillna(0)
        
        return signals
    
    def backtest_strategy(self, coint_result, signals, transaction_cost=0.001):
        """
        Executa backtest da estratégia de pair trading
        
        Args:
            coint_result (dict): Resultado do teste de cointegração
            signals (pd.DataFrame): Sinais de trading
            transaction_cost (float): Custo de transação (% por trade)
            
        Returns:
            dict: Resultados do backtest
        """
        price1 = coint_result['price1']
        price2 = coint_result['price2']
        
        # Retornos dos ativos
        returns1 = price1.pct_change().fillna(0)
        returns2 = price2.pct_change().fillna(0)
        
        # Estratégia: position = 1 significa long asset2, short asset1
        # position = -1 significa short asset2, long asset1
        strategy_returns = []
        
        for i in range(len(signals)):
            position = signals['position'].iloc[i]
            
            if position != 0:
                # Pair trading return: long asset2 - short asset1 (ou vice-versa)
                if position > 0:
                    ret = returns2.iloc[i] - returns1.iloc[i]
                else:
                    ret = returns1.iloc[i] - returns2.iloc[i]
                
                # Aplicar custo de transação nas mudanças de posição
                if i > 0 and signals['position'].iloc[i] != signals['position'].iloc[i-1]:
                    ret -= transaction_cost
                
                strategy_returns.append(ret)
            else:
                strategy_returns.append(0)
        
        strategy_returns = pd.Series(strategy_returns, index=signals.index)
        
        # Métricas de performance
        cumulative_returns = (1 + strategy_returns).cumprod()
        total_return = cumulative_returns.iloc[-1] - 1
        
        # Sharpe ratio anualizado
        annual_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
        annual_volatility = strategy_returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
        
        # Drawdown
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns / running_max - 1)
        max_drawdown = drawdown.min()
        
        # Número de trades
        position_changes = (signals['position'].diff() != 0).sum()
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': position_changes,
            'cumulative_returns': cumulative_returns,
            'strategy_returns': strategy_returns,
            'win_rate': (strategy_returns > 0).mean()
        }
    
    def plot_pair_analysis(self, coint_result, signals, backtest_result):
        """Cria visualização completa da análise do par"""
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=(
                f"Preços Normalizados: {coint_result['asset1']} vs {coint_result['asset2']}",
                f"Spread e Z-Score",
                f"Sinais de Trading",
                f"Performance da Estratégia"
            ),
            specs=[[{"secondary_y": False}],
                   [{"secondary_y": True}],
                   [{"secondary_y": False}],
                   [{"secondary_y": False}]]
        )
        
        # 1. Preços normalizados
        price1_norm = coint_result['price1'] / coint_result['price1'].iloc[0]
        price2_norm = coint_result['price2'] / coint_result['price2'].iloc[0]
        
        fig.add_trace(go.Scatter(
            x=price1_norm.index, y=price1_norm.values,
            name=coint_result['asset1'], line=dict(color='blue')
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=price2_norm.index, y=price2_norm.values,
            name=coint_result['asset2'], line=dict(color='red')
        ), row=1, col=1)
        
        # 2. Spread e Z-Score
        fig.add_trace(go.Scatter(
            x=signals.index, y=signals['spread'],
            name='Spread', line=dict(color='green')
        ), row=2, col=1)
        
        fig.add_trace(go.Scatter(
            x=signals.index, y=signals['z_score'],
            name='Z-Score', line=dict(color='orange')
        ), row=2, col=1)
        
        # Linhas de threshold
        fig.add_hline(y=2, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=-2, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=0, line_dash="dot", line_color="black", row=2, col=1)
        
        # 3. Sinais de trading
        buy_signals = signals[signals['signal'] == 1]
        sell_signals = signals[signals['signal'] == -1]
        
        if len(buy_signals) > 0:
            fig.add_trace(go.Scatter(
                x=buy_signals.index, y=buy_signals['z_score'],
                mode='markers', name='Buy Signal',
                marker=dict(color='green', size=10, symbol='triangle-up')
            ), row=3, col=1)
        
        if len(sell_signals) > 0:
            fig.add_trace(go.Scatter(
                x=sell_signals.index, y=sell_signals['z_score'],
                mode='markers', name='Sell Signal',
                marker=dict(color='red', size=10, symbol='triangle-down')
            ), row=3, col=1)
        
        fig.add_trace(go.Scatter(
            x=signals.index, y=signals['z_score'],
            name='Z-Score', line=dict(color='orange')
        ), row=3, col=1)
        
        # 4. Performance cumulativa
        fig.add_trace(go.Scatter(
            x=backtest_result['cumulative_returns'].index,
            y=(backtest_result['cumulative_returns'] - 1) * 100,
            name='Retorno Cumulativo (%)', line=dict(color='purple')
        ), row=4, col=1)
        
        fig.update_layout(
            title=f"📊 Análise Completa Pair Trading: {coint_result['asset1']} vs {coint_result['asset2']}",
            height=1000,
            showlegend=True
        )
        
        return fig
    
    def plot_correlation_heatmap(self):
        """Cria heatmap de correlação interativo"""
        if self.correlation_matrix is None:
            return None
            
        fig = go.Figure(data=go.Heatmap(
            z=self.correlation_matrix.values,
            x=self.correlation_matrix.columns,
            y=self.correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(self.correlation_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 8},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='🔗 Matriz de Correlação para Pair Trading',
            height=600,
            xaxis_title='Ativos',
            yaxis_title='Ativos'
        )
        
        return fig
    
    def get_best_pairs(self, top_n=5):
        """
        Retorna os melhores pares para trading
        
        Args:
            top_n (int): Número de pares para retornar
            
        Returns:
            list: Lista dos melhores pares ordenados por score
        """
        if not self.selected_pairs:
            return []
        
        # Score baseado em correlação e cointegração
        scored_pairs = []
        
        for pair in self.selected_pairs[:top_n*2]:  # Testar mais pares
            asset1, asset2 = pair['asset1'], pair['asset2']
            coint_result = self.test_cointegration(asset1, asset2)
            
            if coint_result and coint_result['is_cointegrated']:
                # Score baseado em correlação e significância da cointegração
                score = abs(pair['correlation']) * (1 - coint_result['p_value'])
                
                scored_pairs.append({
                    'asset1': asset1,
                    'asset2': asset2,
                    'correlation': pair['correlation'],
                    'p_value': coint_result['p_value'],
                    'score': score,
                    'coint_result': coint_result
                })
        
        # Ordenar por score
        scored_pairs = sorted(scored_pairs, key=lambda x: x['score'], reverse=True)
        
        return scored_pairs[:top_n]
