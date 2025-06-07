import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from data_fetch import ASSET_CATEGORIES

class PortfolioAllocationManager:
    """Gestão de alocação de capital por setor de ativos"""
    
    def __init__(self, price_data, returns_data):
        """
        Inicializa o gestor de alocação
        
        Args:
            price_data (pd.DataFrame): DataFrame com preços dos ativos
            returns_data (pd.DataFrame): DataFrame com retornos dos ativos
        """
        self.price_data = price_data
        self.returns_data = returns_data
        self.asset_categories = ASSET_CATEGORIES
        self.sector_allocations = {}
        self.portfolio_weights = {}
        
    def set_sector_allocation(self, sector_budgets):
        """
        Define alocação de capital por setor
        
        Args:
            sector_budgets (dict): Dicionário com orçamento por setor
                Exemplo: {'acoes': 1000, 'forex': 100, 'criptomoedas': 100}
        """
        self.sector_allocations = sector_budgets
        total_capital = sum(sector_budgets.values())
        
        # Calcular pesos por setor
        sector_weights = {}
        for sector, budget in sector_budgets.items():
            sector_weights[sector] = budget / total_capital
            
        self.sector_weights = sector_weights
        return self
    
    def calculate_portfolio_weights(self, selected_assets, allocation_method='equal_weight'):
        """
        Calcula pesos do portfólio baseado na alocação por setor
        
        Args:
            selected_assets (list): Lista de ativos selecionados
            allocation_method (str): Método de alocação ('equal_weight', 'market_cap', 'risk_parity')
            
        Returns:
            dict: Pesos dos ativos no portfólio
        """
        portfolio_weights = {}
        
        # Categorizar ativos selecionados
        selected_by_sector = {}
        for sector, assets in self.asset_categories.items():
            selected_by_sector[sector] = [asset for asset in selected_assets if asset in assets]
        
        # Calcular pesos para cada setor
        for sector, assets in selected_by_sector.items():
            if not assets or sector not in self.sector_weights:
                continue
                
            sector_weight = self.sector_weights[sector]
            
            if allocation_method == 'equal_weight':
                # Peso igual dentro do setor
                asset_weight = sector_weight / len(assets)
                for asset in assets:
                    portfolio_weights[asset] = asset_weight
                    
            elif allocation_method == 'volatility_parity':
                # Peso baseado no inverso da volatilidade
                volatilities = {}
                for asset in assets:
                    if asset in self.returns_data.columns:
                        vol = self.returns_data[asset].std()
                        volatilities[asset] = 1 / vol if vol > 0 else 1
                
                if volatilities:
                    total_inv_vol = sum(volatilities.values())
                    for asset in assets:
                        inv_vol_weight = volatilities.get(asset, 1) / total_inv_vol
                        portfolio_weights[asset] = sector_weight * inv_vol_weight
                        
            elif allocation_method == 'momentum':
                # Peso baseado no momentum (retorno dos últimos 3 meses)
                momentum_scores = {}
                for asset in assets:
                    if asset in self.returns_data.columns:
                        momentum = self.returns_data[asset].rolling(63).mean().iloc[-1]  # ~3 meses
                        momentum_scores[asset] = max(momentum, 0.001)  # Evitar valores negativos
                
                if momentum_scores:
                    total_momentum = sum(momentum_scores.values())
                    for asset in assets:
                        momentum_weight = momentum_scores.get(asset, 0.001) / total_momentum
                        portfolio_weights[asset] = sector_weight * momentum_weight
        
        self.portfolio_weights = portfolio_weights
        return portfolio_weights
    
    def calculate_sector_performance(self, selected_assets, weights=None):
        """
        Calcula performance por setor
        
        Args:
            selected_assets (list): Lista de ativos selecionados
            weights (dict): Pesos dos ativos (opcional)
            
        Returns:
            dict: Performance por setor
        """
        if weights is None:
            weights = self.portfolio_weights
            
        sector_performance = {}
        
        # Categorizar ativos selecionados
        selected_by_sector = {}
        for sector, assets in self.asset_categories.items():
            selected_by_sector[sector] = [asset for asset in selected_assets if asset in assets]
        
        # Calcular performance para cada setor
        for sector, assets in selected_by_sector.items():
            if not assets:
                continue
                
            sector_returns = []
            sector_weights = []
            
            for asset in assets:
                if asset in self.returns_data.columns and asset in weights:
                    asset_returns = self.returns_data[asset]
                    asset_weight = weights[asset]
                    
                    # Normalizar peso dentro do setor
                    sector_total_weight = sum([weights.get(a, 0) for a in assets])
                    if sector_total_weight > 0:
                        normalized_weight = asset_weight / sector_total_weight
                        sector_returns.append(asset_returns * normalized_weight)
            
            if sector_returns:
                sector_return_series = sum(sector_returns)
                
                # Calcular métricas do setor
                total_return = (1 + sector_return_series).prod() - 1
                annual_return = (1 + total_return) ** (252 / len(sector_return_series)) - 1
                volatility = sector_return_series.std() * np.sqrt(252)
                sharpe = annual_return / volatility if volatility > 0 else 0
                
                # Drawdown
                cum_returns = (1 + sector_return_series).cumprod()
                running_max = cum_returns.expanding().max()
                drawdown = (cum_returns / running_max - 1).min()
                
                sector_performance[sector] = {
                    'total_return': total_return,
                    'annual_return': annual_return,
                    'volatility': volatility,
                    'sharpe_ratio': sharpe,
                    'max_drawdown': drawdown,
                    'cum_returns': cum_returns,
                    'assets': assets,
                    'allocation': self.sector_allocations.get(sector, 0)
                }
        
        return sector_performance
    
    def plot_sector_allocation(self):
        """Cria gráfico de alocação por setor"""
        if not self.sector_allocations:
            return None
            
        sectors = list(self.sector_allocations.keys())
        values = list(self.sector_allocations.values())
        
        # Traduzir nomes dos setores para português
        sector_names = {
            'acoes': 'Ações',
            'etfs_indices': 'ETFs/Índices',
            'criptomoedas': 'Criptomoedas',
            'forex': 'Forex'
        }
        
        labels = [sector_names.get(sector, sector) for sector in sectors]
        
        fig = go.Figure(data=go.Pie(
            labels=labels,
            values=values,
            hole=0.3,
            textinfo='label+percent+value',
            texttemplate='<b>%{label}</b><br>R$ %{value:,.0f}<br>%{percent}',
            marker=dict(colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ))
        
        fig.update_layout(
            title='💰 Alocação de Capital por Setor',
            font=dict(size=12),
            height=500
        )
        
        return fig
    
    def plot_sector_performance_comparison(self, sector_performance):
        """Compara performance entre setores"""
        if not sector_performance:
            return None
            
        # Preparar dados
        sectors = list(sector_performance.keys())
        sector_names = {
            'acoes': 'Ações',
            'etfs_indices': 'ETFs/Índices', 
            'criptomoedas': 'Criptomoedas',
            'forex': 'Forex'
        }
        
        # Métricas para comparação
        metrics_data = {
            'Setor': [sector_names.get(sector, sector) for sector in sectors],
            'Retorno Anual (%)': [sector_performance[sector]['annual_return'] * 100 for sector in sectors],
            'Volatilidade (%)': [sector_performance[sector]['volatility'] * 100 for sector in sectors],
            'Sharpe Ratio': [sector_performance[sector]['sharpe_ratio'] for sector in sectors],
            'Max Drawdown (%)': [sector_performance[sector]['max_drawdown'] * 100 for sector in sectors],
            'Alocação (R$)': [sector_performance[sector]['allocation'] for sector in sectors]
        }
        
        df_metrics = pd.DataFrame(metrics_data)
        
        # Criar subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Retorno vs Volatilidade', 'Sharpe Ratio por Setor', 
                          'Max Drawdown por Setor', 'Alocação de Capital'),
            specs=[[{"type": "scatter"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # 1. Scatter Retorno vs Volatilidade
        fig.add_trace(go.Scatter(
            x=df_metrics['Volatilidade (%)'],
            y=df_metrics['Retorno Anual (%)'],
            mode='markers+text',
            text=df_metrics['Setor'],
            textposition='top center',
            marker=dict(size=15, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']),
            name='Setores'
        ), row=1, col=1)
        
        # 2. Sharpe Ratio
        fig.add_trace(go.Bar(
            x=df_metrics['Setor'],
            y=df_metrics['Sharpe Ratio'],
            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'],
            name='Sharpe Ratio'
        ), row=1, col=2)
        
        # 3. Max Drawdown
        fig.add_trace(go.Bar(
            x=df_metrics['Setor'],
            y=df_metrics['Max Drawdown (%)'],
            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'],
            name='Max Drawdown'
        ), row=2, col=1)
        
        # 4. Alocação
        fig.add_trace(go.Bar(
            x=df_metrics['Setor'],
            y=df_metrics['Alocação (R$)'],
            marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'],
            name='Alocação'
        ), row=2, col=2)
        
        fig.update_layout(
            title='📊 Comparação de Performance por Setor',
            height=800,
            showlegend=False
        )
        
        return fig
    
    def plot_sector_evolution(self, sector_performance):
        """Evolução temporal da performance por setor"""
        if not sector_performance:
            return None
            
        fig = go.Figure()
        
        sector_names = {
            'acoes': 'Ações',
            'etfs_indices': 'ETFs/Índices',
            'criptomoedas': 'Criptomoedas', 
            'forex': 'Forex'
        }
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        for i, (sector, data) in enumerate(sector_performance.items()):
            if 'cum_returns' in data:
                # Converter retorno cumulativo em equity baseado na alocação
                initial_capital = self.sector_allocations.get(sector, 1000)
                equity_curve = data['cum_returns'] * initial_capital
                
                fig.add_trace(go.Scatter(
                    x=equity_curve.index,
                    y=equity_curve.values,
                    mode='lines',
                    name=sector_names.get(sector, sector),
                    line=dict(color=colors[i % len(colors)], width=2),
                    hovertemplate=f'<b>{sector_names.get(sector, sector)}</b><br>' +
                                 'Data: %{x}<br>Equity: R$ %{y:,.2f}<extra></extra>'
                ))
        
        fig.update_layout(
            title='📈 Evolução do Capital por Setor',
            xaxis_title='Data',
            yaxis_title='Equity (R$)',
            hovermode='x unified',
            height=500
        )
        
        return fig
    
    def get_sector_summary(self, sector_performance):
        """Retorna resumo das métricas por setor"""
        if not sector_performance:
            return {}
            
        summary = {}
        
        for sector, data in sector_performance.items():
            sector_name = {
                'acoes': 'Ações',
                'etfs_indices': 'ETFs/Índices',
                'criptomoedas': 'Criptomoedas',
                'forex': 'Forex'
            }.get(sector, sector)
            
            summary[sector_name] = {
                'Alocação': f"R$ {data['allocation']:,.2f}",
                'Retorno Total': f"{data['total_return']:.2%}",
                'Retorno Anual': f"{data['annual_return']:.2%}",
                'Volatilidade': f"{data['volatility']:.2%}",
                'Sharpe Ratio': f"{data['sharpe_ratio']:.2f}",
                'Max Drawdown': f"{data['max_drawdown']:.2%}",
                'Ativos': ', '.join([asset.replace('.SA', '').replace('-USD', '').replace('=X', '') 
                                   for asset in data['assets']])
            }
        
        return summary
