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
        """        # Verificar se temos dados suficientes
        min_periods = min_years * 252  # aproximadamente 252 dias úteis por ano
        if len(self.price_data) < min_periods:
            st.warning(f"Dados insuficientes. Necessário pelo menos {min_years} anos de dados.")
            return []
        
        # Verificar se temos dados de retorno válidos
        if self.returns_data is None or self.returns_data.empty:
            st.error("Dados de retorno inválidos ou vazios.")
            return []
        
        # Calcular matriz de correlação
        try:
            self.correlation_matrix = self.returns_data.corr()
            
            # Verificar se a matriz de correlação foi calculada corretamente
            if self.correlation_matrix is None or self.correlation_matrix.empty:
                st.error("Não foi possível calcular a matriz de correlação.")
                return []
                
        except Exception as e:
            st.error(f"Erro ao calcular matriz de correlação: {str(e)}")
            return []
          # Encontrar pares correlacionados
        pairs = []
        assets = self.correlation_matrix.columns
        
        for i in range(len(assets)):
            for j in range(i+1, len(assets)):
                asset1, asset2 = assets[i], assets[j]
                
                try:
                    # Verificar se os ativos existem na matriz de correlação
                    if asset1 not in self.correlation_matrix.index or asset2 not in self.correlation_matrix.columns:
                        continue
                        
                    correlation = self.correlation_matrix.loc[asset1, asset2]
                    
                    # Verificar se a correlação é um número válido
                    if pd.isna(correlation) or not isinstance(correlation, (int, float)):
                        continue
                        
                    if abs(correlation) >= min_correlation:
                        pairs.append({
                            'asset1': asset1,
                            'asset2': asset2,
                            'correlation': correlation,
                            'abs_correlation': abs(correlation)
                        })
                        
                except Exception as e:
                    st.warning(f"Erro ao processar correlação entre {asset1} e {asset2}: {str(e)}")
                    continue
        
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
                "Spread e Z-Score",
                "Sinais de Trading",
                "Performance da Estratégia"
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
        if self.correlation_matrix is None or self.correlation_matrix.empty:
            try:
                self.correlation_matrix = self.returns_data.corr()
                if self.correlation_matrix is None or self.correlation_matrix.empty:
                    st.error("Não foi possível calcular a matriz de correlação.")
                    return None
            except Exception as e:
                st.error(f"Erro ao calcular matriz de correlação: {str(e)}")
                return None
        
        try:
            fig = go.Figure(data=go.Heatmap(
                z=self.correlation_matrix.values,
                x=self.correlation_matrix.columns,
                y=self.correlation_matrix.columns,
                colorscale='RdBu',
                zmid=0,
                text=np.round(self.correlation_matrix.values, 2),
                texttemplate='%{text}',
                textfont={"size": 8},
                hoverongaps=False            ))
            
            fig.update_layout(
                title='🔗 Matriz de Correlação para Pair Trading',
                height=600,
                xaxis_title='Ativos',
                yaxis_title='Ativos'
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Erro ao criar heatmap de correlação: {str(e)}")
            return None
    
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
    
    def plot_spread_and_signals(self, asset1=None, asset2=None, entry_threshold=2.0, exit_threshold=0.5):
        """
        Plota o spread e os sinais de trading para um par de ativos.
        """
        # Gerar sinais
        if asset1 is None:
            asset1 = self.price_data.columns[0]
        if asset2 is None:
            asset2 = self.price_data.columns[1]
        signals = self.generate_trading_signals(self.cointegration_results[f"{asset1}_{asset2}"], entry_threshold, exit_threshold)
        pair_key = f"{asset1}_{asset2}"
        coint_result = self.cointegration_results.get(pair_key)
        if coint_result is None or signals.empty:
            st.warning(f"Dados insuficientes para plotar {asset1} vs {asset2}")
            return None

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(
                f"Z-Score do Spread: {asset1} vs {asset2}",
                "Spread Bruto"
            ),
            shared_xaxes=True
        )
        # Z-Score
        fig.add_trace(go.Scatter(
            x=signals.index, y=signals['z_score'],
            name='Z-Score', line=dict(color='blue')
        ), row=1, col=1)
        # Spread
        fig.add_trace(go.Scatter(
            x=signals.index, y=signals['spread'],
            name='Spread', line=dict(color='green')
        ), row=2, col=1)
        # Adicionar linhas de threshold
        fig.add_hline(y=entry_threshold, line_dash="dash", line_color="red", row=1, col=1)
        fig.add_hline(y=-entry_threshold, line_dash="dash", line_color="red", row=1, col=1)
        fig.add_hline(y=exit_threshold, line_dash="dot", line_color="green", row=1, col=1)
        fig.add_hline(y=-exit_threshold, line_dash="dot", line_color="green", row=1, col=1)
        fig.add_hline(y=0, line_dash="solid", line_color="black", row=1, col=1)
        fig.add_hline(y=0, line_dash="solid", line_color="black", row=2, col=1)
        # Marcar sinais de compra/venda
        buy_signals = signals[signals['signal'] == 1]
        sell_signals = signals[signals['signal'] == -1]
        if len(buy_signals) > 0:
            fig.add_trace(go.Scatter(
                x=buy_signals.index, y=buy_signals['z_score'],
                mode='markers', name=f'Comprar {asset2}/Vender {asset1}',
                marker=dict(color='green', size=10, symbol='triangle-up')
            ), row=1, col=1)
        if len(sell_signals) > 0:
            fig.add_trace(go.Scatter(
                x=sell_signals.index, y=sell_signals['z_score'],
                mode='markers', name=f'Vender {asset2}/Comprar {asset1}',
                marker=dict(color='red', size=10, symbol='triangle-down')
            ), row=1, col=1)
        fig.update_layout(
            title=f"Análise de Spread e Sinais: {asset1} vs {asset2}",
            height=700, width=900,
            showlegend=True,
            xaxis2_title="Data",
            yaxis_title="Z-Score",
            yaxis2_title="Spread"
        )
        return fig

    def plot_strategy_performance(self, asset1=None, asset2=None):
        """
        Plota a performance da estratégia de pair trading para um par de ativos.
        """
        if asset1 is None:
            asset1 = self.price_data.columns[0]
        if asset2 is None:
            asset2 = self.price_data.columns[1]
        signals = self.generate_trading_signals(self.cointegration_results[f"{asset1}_{asset2}"])
        pair_key = f"{asset1}_{asset2}"
        coint_result = self.cointegration_results.get(pair_key)
        if coint_result is None or signals.empty:
            st.warning(f"Dados insuficientes para plotar performance de {asset1} vs {asset2}")
            return None
        backtest_result = self.backtest_strategy(coint_result, signals)
        if backtest_result is None:
            return None
        portfolio = backtest_result['cumulative_returns']
        price1 = coint_result['price1']
        price2 = coint_result['price2']
        price1_norm = price1 / price1.iloc[0]
        price2_norm = price2 / price2.iloc[0]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=price1_norm.index, y=price1_norm.values,
            name=asset1, line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=price2_norm.index, y=price2_norm.values,
            name=asset2, line=dict(color='red')
        ))
        fig.add_trace(go.Scatter(
            x=portfolio.index, y=portfolio.values,
            name='Pair Trading Strategy', line=dict(color='green', width=2)
        ))
        # Adicionar métricas como anotações
        annotations = [
            f"Retorno Total: {backtest_result['total_return']:.2%}",
            f"Sharpe Ratio: {backtest_result['sharpe_ratio']:.2f}",
            f"Máximo Drawdown: {backtest_result['max_drawdown']:.2%}",
            f"Número de Trades: {backtest_result['num_trades']}",
            f"Win Rate: {backtest_result['win_rate']:.2%}"
        ]
        annotation_text = "<br>".join(annotations)
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.02, y=0.98,
            text=annotation_text,
            showarrow=False,
            font=dict(size=12, color="black"),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            opacity=0.8
        )
        fig.update_layout(
            title=f"Performance da Estratégia: {asset1} vs {asset2}",
            height=600, width=900,
            xaxis_title="Data",
            yaxis_title="Retorno Normalizado (Início = 1)",
            legend=dict(x=0.02, y=0.02, bgcolor="white")
        )
        return fig

    def analyze_pair(self, asset1, asset2, lookback_period=90, z_threshold=2.0):
        """
        Analisa um par específico de ativos para pair trading
        
        Args:
            asset1 (str): Nome do primeiro ativo
            asset2 (str): Nome do segundo ativo
            lookback_period (int): Período de lookback em dias
            z_threshold (float): Threshold para sinais de trading
              Returns:
            dict: Resultados da análise
        """
        try:
            # Verificar se os ativos existem
            if asset1 not in self.price_data.columns or asset2 not in self.price_data.columns:
                return None
            
            # Calcular correlação
            if self.correlation_matrix is None:
                try:
                    self.correlation_matrix = self.returns_data.corr()
                    if self.correlation_matrix is None or self.correlation_matrix.empty:
                        st.error("Erro ao calcular matriz de correlação.")
                        return None
                except Exception as e:
                    st.error(f"Erro ao calcular matriz de correlação: {str(e)}")
                    return None
            
            # Verificar se os ativos existem na matriz de correlação
            if (asset1 not in self.correlation_matrix.index or 
                asset2 not in self.correlation_matrix.columns):
                st.error(f"Ativos {asset1} ou {asset2} não encontrados na matriz de correlação.")
                return None
            
            try:
                correlation = self.correlation_matrix.loc[asset1, asset2]
                if pd.isna(correlation):
                    st.error(f"Correlação entre {asset1} e {asset2} não disponível.")
                    return None
            except Exception as e:
                st.error(f"Erro ao acessar correlação entre {asset1} e {asset2}: {str(e)}")
                return None
            
            # Teste de cointegração
            coint_result = self.test_cointegration(asset1, asset2)
            
            if coint_result is None:
                return None
            
            # Gerar sinais de trading
            signals = self.generate_trading_signals(
                coint_result,
                entry_threshold=z_threshold,
                exit_threshold=z_threshold/2
            )
            
            # Contar sinais
            buy_signals = len(signals[signals['signal'] == 1])
            sell_signals = len(signals[signals['signal'] == -1])
            total_signals = buy_signals + sell_signals
            
            # Executar backtest básico
            backtest_result = self.backtest_strategy(coint_result, signals)
            
            return {
                'correlation': correlation,
                'cointegration': '✅ Sim' if coint_result.get('is_cointegrated', False) else '❌ Não',
                'p_value': coint_result.get('p_value', 1.0),
                'signals': total_signals,
                'buy_signals': buy_signals,
                'sell_signals': sell_signals,
                'sharpe_ratio': backtest_result.get('sharpe_ratio', 0) if backtest_result else 0,
                'total_return': backtest_result.get('total_return', 0) if backtest_result else 0,
                'coint_result': coint_result,
                'trading_signals': signals,
                'backtest': backtest_result
            }
            
        except Exception as e:
            st.error(f"Erro na análise do par {asset1}/{asset2}: {str(e)}")
            return None
