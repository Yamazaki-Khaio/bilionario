"""
Módulo auxiliar para gestão por setor
Reduz complexidade cognitiva extraindo funcionalidades em funções menores
"""

import pandas as pd
import numpy as np
import plotly.express as px
import streamlit as st
import json
import io
from datetime import datetime
from portfolio_allocation import PortfolioAllocationManager
from data_fetch import ASSET_CATEGORIES
from financial_formatting import format_percentage, format_currency
from constants import MATERIALS_SECTOR, TELECOMMUNICATIONS_SECTOR, TECHNOLOGY_SECTOR


def get_risk_profile_config(risk_tolerance):
    """Retorna configuração de perfil de risco e sugestões de alocação"""
    if risk_tolerance <= 3:  # Conservador
        return {
            'suggested_allocation': {
                'financeiro': 30, 'energia': 20, 'consumo': 15, 
                MATERIALS_SECTOR: 10, TELECOMMUNICATIONS_SECTOR: 10, TECHNOLOGY_SECTOR: 15
            },
            'profile': "Conservador",
            'color': "green"
        }
    elif risk_tolerance <= 7:  # Moderado
        return {
            'suggested_allocation': {
                'financeiro': 25, 'energia': 15, 'consumo': 20, 
                MATERIALS_SECTOR: 15, TELECOMMUNICATIONS_SECTOR: 10, TECHNOLOGY_SECTOR: 15
            },
            'profile': "Moderado",
            'color': "orange"
        }
    else:  # Agressivo
        return {
            'suggested_allocation': {
                'financeiro': 20, 'energia': 10, 'consumo': 15, 
                MATERIALS_SECTOR: 20, TELECOMMUNICATIONS_SECTOR: 15, TECHNOLOGY_SECTOR: 20
            },
            'profile': "Agressivo",
            'color': "red"
        }


def display_risk_profile(profile_color, risk_profile):
    """Exibe o perfil de risco com cor apropriada"""
    message = f"📊 **Perfil de Risco**: {risk_profile} | **Sugestão**: Baseada em sua tolerância ao risco"
    
    if profile_color == "green":
        st.success(message)
    elif profile_color == "orange":
        st.warning(message)
    else:
        st.error(message)


def configure_sector_allocation(available_sectors, suggested_allocation):
    """Configura alocação manual por setor"""
    sector_allocations = {}
    total_allocation = 0
    
    use_suggestion = st.checkbox("🎯 Usar Alocação Sugerida", value=True)
    
    for sector in available_sectors:
        if use_suggestion and sector in suggested_allocation:
            default_value = suggested_allocation[sector]
        else:
            default_value = 100 // len(available_sectors)
            
        allocation = st.slider(
            f"{sector.title()} (%)",
            min_value=0,
            max_value=100,
            value=default_value,
            key=f"allocation_{sector}"
        )
        sector_allocations[sector] = allocation
        total_allocation += allocation
    
    return sector_allocations, total_allocation


def display_allocation_status(total_allocation, total_capital, sector_allocations):
    """Exibe status da alocação e valores por setor"""
    if total_allocation == 100:
        st.success("✅ Alocação Balanceada!")
    elif total_allocation < 100:
        st.warning(f"⚠️ Faltam {100-total_allocation}%")
    else:
        st.error(f"❌ Excesso de {total_allocation-100}%")
    
    st.metric("Total Alocado", f"{total_allocation}%")
    
    # Mostrar valor por setor
    st.markdown("**Valores por Setor:**")
    for sector, pct in sector_allocations.items():
        value = total_capital * (pct / 100)
        if pct > 0:
            st.write(f"• {sector.title()}: R$ {value:,.0f}")


def create_allocation_visualizations(sector_allocations, available_sectors, suggested_allocation):
    """Cria visualizações da alocação"""
    tab1, tab2, tab3 = st.tabs(["🥧 Pizza", "📊 Barras", "🎯 Comparação"])
    
    # Filtrar setores com alocação > 0
    active_sectors = {k: v for k, v in sector_allocations.items() if v > 0}
    
    with tab1:
        if active_sectors:
            fig_pie = px.pie(
                values=list(active_sectors.values()),
                names=[s.title() for s in active_sectors.keys()],
                title='Distribuição por Setor',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    
    with tab2:
        if active_sectors:
            fig_bar = px.bar(
                x=list(active_sectors.values()),
                y=[s.title() for s in active_sectors.keys()],
                title='Alocação por Setor (%)',
                orientation='h',
                color=list(active_sectors.values()),
                color_continuous_scale="viridis"
            )
            st.plotly_chart(fig_bar, use_container_width=True)
    
    with tab3:
        # Comparar com alocação sugerida
        comparison_df = pd.DataFrame({
            'Setor': [s.title() for s in available_sectors],
            'Sua Alocação (%)': [sector_allocations.get(s, 0) for s in available_sectors],
            'Sugerida (%)': [suggested_allocation.get(s, 0) for s in available_sectors]
        })
        
        fig_comparison = px.bar(
            comparison_df,
            x='Setor',
            y=['Sua Alocação (%)', 'Sugerida (%)'],
            title='Comparação: Sua Alocação vs Sugerida',
            barmode='group'
        )
        st.plotly_chart(fig_comparison, use_container_width=True)


def analyze_sector_performance(df, returns, sector_allocations):
    """Executa análise de performance por setor"""
    try:
        # Inicializar gestor de alocação
        allocation_manager = PortfolioAllocationManager(df, returns)
        
        # Configurar alocação por setor
        sector_budgets = {sector: pct/100 for sector, pct in sector_allocations.items() if pct > 0}
        allocation_manager.set_sector_allocation(sector_budgets)
        
        # Selecionar ativos por setor
        selected_assets = []
        for sector in sector_budgets.keys():
            if sector in ASSET_CATEGORIES:
                sector_assets = [asset for asset in ASSET_CATEGORIES[sector] if asset in df.columns]
                selected_assets.extend(sector_assets[:3])  # Top 3 por setor
        
        if len(selected_assets) >= 3:
            # Calcular pesos do portfolio
            allocation_method = st.selectbox(
                "Método de Alocação:",
                ["equal_weight", "market_cap", "risk_parity"]
            )
            
            portfolio_weights = allocation_manager.calculate_portfolio_weights(
                selected_assets, allocation_method
            )
            
            return allocation_manager, selected_assets, portfolio_weights
        else:
            st.warning("⚠️ Não há ativos suficientes nos setores selecionados para análise")
            return None, None, None
            
    except Exception as e:
        st.error(f"❌ Erro na análise de alocação: {str(e)}")
        return None, None, None


def display_performance_results(allocation_manager, selected_assets, portfolio_weights):
    """Exibe resultados da análise de performance"""
    # Mostrar alocação
    st.subheader("📊 Visualização da Alocação Calculada")
    
    # Gráfico de pizza da alocação
    fig_allocation = allocation_manager.plot_sector_allocation()
    if fig_allocation:
        st.plotly_chart(fig_allocation, use_container_width=True, key="sector_allocation_chart")
    
    # Calcular performance por setor
    sector_performance = allocation_manager.calculate_sector_performance(
        selected_assets, portfolio_weights
    )
    
    if sector_performance:
        # Comparação de performance
        fig_comparison = allocation_manager.plot_sector_performance_comparison(sector_performance)
        if fig_comparison:
            st.plotly_chart(fig_comparison, use_container_width=True, key="sector_performance_comparison")
        
        # Tabela de performance
        st.subheader("📋 Performance Detalhada por Setor")
        performance_df = pd.DataFrame(sector_performance).T
        performance_df = performance_df.round(4)
        st.dataframe(performance_df, use_container_width=True)
        
        # Insights automáticos
        best_sector = performance_df['annual_return'].idxmax()
        worst_sector = performance_df['annual_return'].idxmin()
        
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"🎯 **Melhor Setor**: {best_sector}")
            st.write(f"Retorno: {format_percentage(performance_df.loc[best_sector, 'annual_return'])}")
        with col2:
            st.warning(f"⚠️ **Setor Desafiador**: {worst_sector}")
            st.write(f"Retorno: {format_percentage(performance_df.loc[worst_sector, 'annual_return'])}")


def generate_recommendations(risk_tolerance, sector_allocations, rebalance_frequency):
    """Gera recomendações personalizadas"""
    recommendations = []
    
    # Baseado no perfil de risco
    if risk_tolerance <= 3:
        recommendations.append("🛡️ **Conservador**: Considere aumentar alocação em setores defensivos (energia, financeiro)")
        recommendations.append("📊 Mantenha diversificação para reduzir volatilidade")
    elif risk_tolerance <= 7:
        recommendations.append("⚖️ **Moderado**: Balance entre crescimento e estabilidade")
        recommendations.append("🔄 Rebalanceie trimestralmente para manter alocação alvo")
    else:
        recommendations.append("🚀 **Agressivo**: Foque em setores de crescimento (tecnologia, materiais)")
        recommendations.append("📈 Aceite maior volatilidade para potencial maior retorno")
    
    # Baseado na alocação atual
    max_allocation = max(sector_allocations.values()) if sector_allocations.values() else 0
    if max_allocation > 40:
        recommendations.append("⚠️ **Concentração Alta**: Considere diversificar mais entre setores")
    
    if len([v for v in sector_allocations.values() if v > 0]) < 3:
        recommendations.append("🎯 **Diversificação**: Inclua pelo menos 3-4 setores diferentes")
    
    recommendations.append(f"📅 **Rebalanceamento**: Revise sua alocação {rebalance_frequency.lower()}")
    recommendations.append("📊 **Monitoramento**: Acompanhe performance relativa entre setores")
    
    return recommendations


def save_configuration(sector_allocations, total_capital, risk_tolerance, rebalance_frequency):
    """Salva configuração em arquivo JSON"""
    config = {
        'sector_allocations': sector_allocations,
        'total_capital': total_capital,
        'risk_tolerance': risk_tolerance,
        'rebalance_frequency': rebalance_frequency,
        'timestamp': datetime.now().isoformat()
    }
    with open('sector_allocation_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    st.success("✅ Configuração salva!")


def export_to_excel(total_allocation, sector_allocations, total_capital, risk_profile, rebalance_frequency):
    """Exporta alocação para arquivo Excel"""
    if total_allocation == 100:
        allocation_data = pd.DataFrame([
            {
                'Setor': sector.title(),
                'Alocação (%)': pct,
                'Valor (R$)': total_capital * (pct / 100),
                'Perfil_Risco': risk_profile,
                'Data_Criacao': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            for sector, pct in sector_allocations.items() if pct > 0
        ])
        
        # Criar arquivo Excel para download
        excel_buffer = io.BytesIO()
        
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            allocation_data.to_excel(writer, sheet_name='Alocação Setorial', index=False)
            
            # Adicionar resumo
            summary_data = pd.DataFrame([{
                'Total_Capital': total_capital,
                'Perfil_Risco': risk_profile,
                'Frequencia_Rebalanceamento': rebalance_frequency,
                'Setores_Alocados': len([pct for pct in sector_allocations.values() if pct > 0]),
                'Concentracao_Maxima': max(sector_allocations.values()) if sector_allocations.values() else 0
            }])
            summary_data.to_excel(writer, sheet_name='Resumo', index=False)
        
        excel_buffer.seek(0)
        
        st.download_button(
            label="📊 Download Alocação Setorial",
            data=excel_buffer.getvalue(),
            file_name=f"alocacao_setorial_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
        
        st.success("✅ Arquivo Excel preparado para download!")
    else:
        st.warning("⚠️ Complete a alocação (100%) antes de exportar")
