import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import streamlit as st

class PCAAdvancedAnalysis:
    """Análise PCA avançada com foco nos 3 primeiros componentes principais"""
    
    def __init__(self, returns_df):
        """
        Inicializa análise PCA avançada
        
        Args:
            returns_df (pd.DataFrame): DataFrame com retornos dos ativos
        """
        self.returns_df = returns_df
        self.scaler = StandardScaler()
        self.pca = None
        self.scaled_returns = None
        self.components = None
        self.loadings = None
        self.explained_variance_ratio = None
        self.asset_names = returns_df.columns.tolist()
        
    def fit_pca(self, n_components=None):
        """Executa análise PCA"""
        if n_components is None:
            n_components = min(len(self.asset_names), 10)
            
        # Padronizar dados
        self.scaled_returns = self.scaler.fit_transform(self.returns_df)
        
        # Executar PCA
        self.pca = PCA(n_components=n_components, random_state=42)
        self.components = self.pca.fit_transform(self.scaled_returns)
        
        # Calcular loadings
        self.loadings = pd.DataFrame(
            self.pca.components_.T,
            columns=[f'PC{i+1}' for i in range(n_components)],
            index=self.asset_names
        )
        
        self.explained_variance_ratio = self.pca.explained_variance_ratio_
        
        return self
    
    def plot_scree_plot(self):
        """Cria scree plot da variância explicada"""
        fig = go.Figure()
        
        # Criar dados para o eixo x (números dos componentes)
        x_values = list(range(1, len(self.explained_variance_ratio) + 1))
        
        # Variância individual
        individual_variance = self.explained_variance_ratio * 100
        fig.add_trace(go.Bar(
            x=x_values,
            y=individual_variance,
            name='Variância Individual',
            marker=dict(color='rgba(58, 71, 80, 0.6)'),
            hovertemplate='<b>PC%{x}</b><br>Variância Explicada: %{y:.2f}%<extra></extra>'
        ))
        
        # Variância cumulativa
        cumulative_var = np.cumsum(self.explained_variance_ratio) * 100
        fig.add_trace(go.Scatter(
            x=x_values,
            y=cumulative_var,
            mode='lines+markers',
            name='Variância Cumulativa',
            line=dict(color='#FF4B4B', width=3),
            marker=dict(size=8),
            hovertemplate='<b>PC%{x}</b><br>Variância Cumulativa: %{y:.2f}%<extra></extra>'
        ))
        
        # Linhas de referência para facilitar interpretação
        fig.add_hline(y=80, line_dash="dot", line_color="green",
                      annotation_text="80% da Variância", annotation_position="top right")
        fig.add_hline(y=90, line_dash="dot", line_color="blue",
                      annotation_text="90% da Variância", annotation_position="top right")
        
        # Adicionar anotações para destacar componentes importantes
        for idx, cumsum in enumerate(cumulative_var):
            if cumsum >= 80 and (idx == 0 or cumulative_var[idx-1] < 80):
                fig.add_annotation(
                    x=idx+1, y=cumsum,
                    text=f"PC{idx+1}: {cumsum:.1f}%",
                    showarrow=True,
                    arrowhead=1,
                    arrowsize=1,
                    arrowcolor="#00CC96",
                    ax=-40,
                    ay=-40
                )
        
        fig.update_layout(
            title={
                'text': '📊 Análise de Componentes Principais - Variância Explicada',
                'y':0.95,
                'x':0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            },
            xaxis_title={
                'text': 'Componente Principal (PC)',
                'font': dict(size=14)
            },
            yaxis_title={
                'text': 'Variância Explicada (%)',
                'font': dict(size=14)
            },
            hovermode='x unified',
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            barmode='group',
            bargap=0.2,
            margin=dict(l=60, r=60, t=80, b=60)
        )
        
        # Adicionar barra de cores para facilitar a interpretação
        fig.update_layout(coloraxis_showscale=True)
        
        return fig
    
    def plot_loadings_analysis(self):
        """Análise detalhada dos loadings dos 3 primeiros componentes"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('PC1 Loadings', 'PC2 Loadings', 'PC3 Loadings', 'Loadings Comparison'),
            specs=[[{"type": "bar"}, {"type": "bar"}], 
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # PC1 Loadings
        pc1_loadings = self.loadings['PC1'].abs().sort_values(ascending=True)
        fig.add_trace(go.Bar(
            y=pc1_loadings.index,
            x=pc1_loadings.values,
            orientation='h',
            name='PC1',
            marker_color='blue'
        ), row=1, col=1)
        
        # PC2 Loadings
        pc2_loadings = self.loadings['PC2'].abs().sort_values(ascending=True)
        fig.add_trace(go.Bar(
            y=pc2_loadings.index,
            x=pc2_loadings.values,
            orientation='h',
            name='PC2',
            marker_color='red'
        ), row=1, col=2)
        
        # PC3 Loadings
        pc3_loadings = self.loadings['PC3'].abs().sort_values(ascending=True)
        fig.add_trace(go.Bar(
            y=pc3_loadings.index,
            x=pc3_loadings.values,
            orientation='h',
            name='PC3',
            marker_color='green'
        ), row=2, col=1)
        
        # Comparação PC1 vs PC2
        fig.add_trace(go.Scatter(
            x=self.loadings['PC1'],
            y=self.loadings['PC2'],
            mode='markers+text',
            text=self.loadings.index,
            textposition='top center',
            name='PC1 vs PC2',
            marker=dict(size=10, color='purple')
        ), row=2, col=2)
        
        fig.update_layout(
            title='🎯 Análise de Loadings - 3 Primeiros Componentes Principais',
            height=800,
            showlegend=False
        )
        
        return fig
    
    def plot_correlation_matrix(self):
        """Matriz de correlação interativa"""
        corr_matrix = self.returns_df.corr()
        
        # Criar heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='🔗 Matriz de Correlação dos Ativos',
            height=600,
            xaxis_title='Ativos',
            yaxis_title='Ativos'
        )
        
        return fig
    
    def plot_component_space_3d(self):
        """Visualização 3D dos 3 primeiros componentes principais"""
        if self.components.shape[1] < 3:
            return None
            
        # Criar cores baseadas no tempo
        colors = np.arange(len(self.components))
        
        fig = go.Figure(data=go.Scatter3d(
            x=self.components[:, 0],
            y=self.components[:, 1],
            z=self.components[:, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=colors,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Período"),
                opacity=0.8
            ),
            text=[f'Dia {i+1}' for i in range(len(self.components))],
            hovertemplate='<b>%{text}</b><br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>PC3: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='🌐 Espaço dos 3 Primeiros Componentes Principais',
            scene=dict(
                xaxis_title=f'PC1 ({self.explained_variance_ratio[0]:.1%})',
                yaxis_title=f'PC2 ({self.explained_variance_ratio[1]:.1%})',
                zaxis_title=f'PC3 ({self.explained_variance_ratio[2]:.1%})'
            ),
            height=600
        )
        
        return fig
    
    def plot_biplot(self):
        """Biplot dos dois primeiros componentes principais"""
        if self.components.shape[1] < 2:
            return None
            
        fig = go.Figure()
        
        # Pontos dos componentes (scores)
        fig.add_trace(go.Scatter(
            x=self.components[:, 0],
            y=self.components[:, 1],
            mode='markers',
            name='Observações',
            marker=dict(size=6, color='lightblue', opacity=0.6),
            hovertemplate='PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>'
        ))
        
        # Vetores dos loadings
        scale_factor = 3  # Para melhor visualização
        for i, asset in enumerate(self.asset_names):
            fig.add_trace(go.Scatter(
                x=[0, self.loadings.loc[asset, 'PC1'] * scale_factor],
                y=[0, self.loadings.loc[asset, 'PC2'] * scale_factor],
                mode='lines+text',
                name=asset,
                line=dict(color='red', width=2),
                text=['', asset],
                textposition='top center',
                showlegend=False
            ))
        
        fig.update_layout(
            title=f'📈 Biplot PC1 vs PC2 (Variância Explicada: {sum(self.explained_variance_ratio[:2]):.1%})',
            xaxis_title=f'PC1 ({self.explained_variance_ratio[0]:.1%})',
            yaxis_title=f'PC2 ({self.explained_variance_ratio[1]:.1%})',
            height=600
        )
        
        return fig
    
    def get_component_interpretation(self):
        """Interpreta os 3 primeiros componentes principais"""
        interpretations = {}
        
        for i in range(min(3, len(self.explained_variance_ratio))):
            pc_name = f'PC{i+1}'
            loadings = self.loadings[pc_name]
            
            # Top 3 ativos com maior loading (absoluto)
            top_assets = loadings.abs().nlargest(3)
            
            # Interpretação baseada nos loadings
            dominant_assets = []
            for asset in top_assets.index:
                loading_val = loadings[asset]
                direction = "positiva" if loading_val > 0 else "negativa"
                dominant_assets.append(f"{asset} ({direction})")
            
            interpretations[pc_name] = {
                'variance_explained': self.explained_variance_ratio[i],
                'cumulative_variance': sum(self.explained_variance_ratio[:i+1]),
                'dominant_assets': dominant_assets,
                'interpretation': f"Representa {self.explained_variance_ratio[i]:.1%} da variância total, dominado por: {', '.join(top_assets.index[:2])}"
            }
        
        return interpretations
    
    def create_summary_metrics(self):
        """Cria métricas resumo da análise PCA"""
        total_components = len(self.explained_variance_ratio)
        variance_80_pct = np.nonzero(np.cumsum(self.explained_variance_ratio) >= 0.8)[0]
        components_for_80pct = variance_80_pct[0] + 1 if len(variance_80_pct) > 0 else total_components
        
        return {
            'total_components': total_components,
            'variance_pc1': self.explained_variance_ratio[0],
            'variance_pc2': self.explained_variance_ratio[1] if len(self.explained_variance_ratio) > 1 else 0,
            'variance_pc3': self.explained_variance_ratio[2] if len(self.explained_variance_ratio) > 2 else 0,
            'variance_first_3': sum(self.explained_variance_ratio[:3]) if len(self.explained_variance_ratio) >= 3 else sum(self.explained_variance_ratio),
            'components_for_80pct': components_for_80pct,
            'total_variance_explained': sum(self.explained_variance_ratio)
        }
    
    def analyze_components(self):
        """Executa análise completa dos componentes principais"""
        try:
            # Executar PCA se ainda não foi feito
            if self.pca is None:
                self.fit_pca()
            
            # Retornar resultados da análise
            return {
                'explained_variance_ratio': self.explained_variance_ratio,
                'loadings': self.loadings,
                'components': self.components,
                'asset_names': self.asset_names
            }
        except Exception as e:
            print(f"Erro na análise de componentes: {e}")
            return None
    
    def plot_scree(self):
        """Alias para plot_scree_plot para compatibilidade"""
        return self.plot_scree_plot()
    
    def plot_loadings_heatmap(self):
        """Alias para plot_loadings_analysis para compatibilidade"""
        return self.plot_loadings_analysis()
    
    def plot_biplot_3d(self):
        """Alias para plot_component_space_3d para compatibilidade"""
        return self.plot_component_space_3d()
    
    def interpret_components(self):
        """Interpreta os componentes principais"""
        try:
            if self.pca is None:
                self.fit_pca()
            
            interpretations = []
            for i in range(min(3, len(self.explained_variance_ratio))):
                # Pegar os loadings do componente
                loadings = self.loadings.iloc[:, i]
                
                # Identificar os ativos mais importantes
                top_assets = loadings.abs().nlargest(5)
                
                # Criar interpretação básica
                interpretation = f"Componente dominado por: {', '.join(top_assets.index[:3])}"
                
                interpretations.append({
                    'variance_explained': self.explained_variance_ratio[i],
                    'interpretation': interpretation,
                    'top_assets': [(asset, loadings[asset]) for asset in top_assets.index]
                })
            
            return interpretations
        except Exception as e:
            print(f"Erro na interpretação: {e}")
            return []
