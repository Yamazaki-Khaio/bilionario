"""
Módulo auxiliar para análise PCA avançada
Reduz complexidade cognitiva extraindo funcionalidades em funções menores
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from financial_formatting import format_percentage, format_ratio
from constants import (
    LOADING_PC1_LABEL,
    LOADING_PC2_LABEL,
    TOTAL_RETURN_LABEL,
    WEIGHT_PERCENTAGE_LABEL,
)


def setup_pca_sidebar(df):
    """Configura sidebar para análise PCA"""
    st.sidebar.subheader("🎯 Configuração PCA")

    selected_assets = st.sidebar.multiselect(
        "Selecione ativos para análise PCA:",
        df.columns.tolist(),
        default=df.columns.tolist()[:10],
    )

    if len(selected_assets) < 3:
        return None, None, None, None, None

    n_components = st.sidebar.slider(
        "Número de Componentes:",
        min_value=2,
        max_value=min(10, len(selected_assets)),
        value=min(5, len(selected_assets)),
    )

    rebalance_freq = st.sidebar.selectbox(
        "Frequência de Rebalanceamento:", ["Mensal", "Trimestral", "Semestral"]
    )

    window_size = st.sidebar.slider("Janela Rolling (dias):", 30, 252, 90)
    rebalance_window = st.sidebar.slider(
        "Janela de Rebalanceamento (dias):", 20, 60, 30
    )

    return selected_assets, n_components, rebalance_freq, window_size, rebalance_window


def execute_static_pca_analysis(returns_selected, selected_assets, n_components):
    """Executa análise PCA estática"""
    # Filtrar apenas os ativos selecionados
    if selected_assets is not None and len(selected_assets) > 0:
        returns_selected = returns_selected[selected_assets]
    
    # Executar PCA
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(returns_selected.fillna(0))
    pca = PCA(n_components=n_components, random_state=42)
    components = pca.fit_transform(scaled_data)

    # Gráfico Scree Plot
    explained_var = pca.explained_variance_ratio_
    fig_scree = px.bar(
        x=list(range(1, len(explained_var) + 1)),
        y=explained_var * 100,
        title="📊 Scree Plot - Variância Explicada por Componente",
        labels={"x": "Componente", "y": "Variância Explicada (%)"},
    )
    st.plotly_chart(fig_scree, use_container_width=True)

    return pca, components, explained_var


def display_pca_loadings(pca, selected_assets, explained_var):
    """Exibe loadings dos componentes principais"""
    st.subheader("🎯 Loadings dos Componentes Principais")

    # Criar DataFrame dos loadings
    loadings_df = pd.DataFrame(
        pca.components_[:2].T,  # Primeiros 2 componentes
        columns=[LOADING_PC1_LABEL, LOADING_PC2_LABEL],
        index=selected_assets,
    )

    # Mostrar loadings em colunas
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Loadings PC1:**")
        loadings_pc1 = loadings_df[LOADING_PC1_LABEL].sort_values(
            key=abs, ascending=False
        )
        st.dataframe(loadings_pc1)

    with col2:
        st.write("**Loadings PC2:**")
        loadings_pc2 = loadings_df[LOADING_PC2_LABEL].sort_values(
            key=abs, ascending=False
        )
        st.dataframe(loadings_pc2)

    # Interpretação dos componentes
    st.subheader("🔍 Interpretação dos Componentes")

    # PC1
    top_pc1 = loadings_pc1.abs().nlargest(3)
    st.write(f"**PC1** (explica {explained_var[0]:.1%} da variância):")
    st.write(f"Dominado por: {', '.join(top_pc1.index[:3])}")

    # PC2
    if len(explained_var) > 1:
        top_pc2 = loadings_pc2.abs().nlargest(3)
        st.write(f"**PC2** (explica {explained_var[1]:.1%} da variância):")
        st.write(f"Dominado por: {', '.join(top_pc2.index[:3])}")


def execute_rolling_pca_analysis(
    returns_selected, selected_assets, window_size, rebalance_freq, rebalance_window
):
    """Executa análise PCA rolling"""
    st.subheader("📈 Análise PCA Rolling")
    st.info("💡 Analisa como os componentes principais evoluem ao longo do tempo")

    try:
        # Filtrar apenas os ativos selecionados
        if selected_assets is not None and len(selected_assets) > 0:
            returns_selected = returns_selected[selected_assets]
            
        # Verificar e converter parâmetros para os tipos corretos
        window_size_int = int(window_size)
        rebalance_window_int = int(rebalance_window)
        # Garantir que rebalance_freq seja uma string
        rebalance_freq = str(rebalance_freq)

        # Calcular PCA rolling
        rolling_variance = []
        rolling_loadings_pc1 = []
        rolling_dates = []
        with st.spinner("Calculando PCA rolling..."):
            for i in range(window_size_int, len(returns_selected)):
                window_data = returns_selected.iloc[i - window_size_int : i]

                # PCA na janela
                scaler_window = StandardScaler()
                scaled_window = scaler_window.fit_transform(window_data.fillna(0))
                pca_window = PCA(n_components=2, random_state=42)
                pca_window.fit(scaled_window)

                rolling_variance.append(pca_window.explained_variance_ratio_[0])
                rolling_loadings_pc1.append(pca_window.components_[0])
                rolling_dates.append(returns_selected.index[i])
            
            # Plot variância explicada rolling
            title_text = "📊 Variância Explicada do PC1 (Janela de " + str(window_size_int) + " dias)"

            fig_rolling_var = px.line(
                x=rolling_dates,
                y=rolling_variance,
                title=title_text,
                labels={"x": "Data", "y": "Variância Explicada PC1"},
            )
            
            # Adicionar marcadores de rebalanceamento
            rebalance_dates = []
            for i in range(0, len(rolling_dates), rebalance_window_int):
                if i < len(rolling_dates):
                    # Garantir que adicionamos apenas datas válidas
                    try:
                        date_value = rolling_dates[i]
                        rebalance_dates.append(date_value)
                    except Exception as e:
                        st.warning(f"Erro ao adicionar data de rebalanceamento no índice {i}: {str(e)}")
            
            # Lista para armazenar formas e anotações
            shapes = []
            annotations = []
            
            # Criar linhas verticais para datas de rebalanceamento
            for rebalance_date in rebalance_dates:
                try:
                    # Converter para timestamp se necessário
                    if not isinstance(rebalance_date, pd.Timestamp):
                        try:
                            rebalance_date = pd.Timestamp(rebalance_date)
                        except Exception as e:
                            st.warning("Erro ao converter data para timestamp: " + str(e) + ". Usando formato original.")
                    
                    # Converter para string no formato YYYY-MM-DD
                    try:
                        date_str = rebalance_date.strftime("%Y-%m-%d")
                    except AttributeError:
                        # Se não for possível usar strftime, tentar converter explicitamente
                        date_str = str(rebalance_date)
                    
                    # Adicionar forma (linha vertical)
                    shapes.append(dict(
                        type="line",
                        x0=date_str,
                        y0=0,
                        x1=date_str,
                        y1=1,
                        line=dict(
                            color="red",
                            width=1,
                            dash="dot",
                        ),
                        xref="x",
                        yref="paper"
                    ))
                    
                    # Adicionar anotação
                    annotations.append(dict(
                        x=date_str,
                        y=1,
                        xref="x",
                        yref="paper",
                        text="Rebalanceamento: " + str(rebalance_freq),
                        showarrow=False,
                        font=dict(color="red", size=10),
                        textangle=-90
                    ))
                except Exception as e:
                    st.warning("Erro ao adicionar linha de rebalanceamento: " + str(e))
            
            # Atualizar layout do gráfico com as formas e anotações
            fig_rolling_var.update_layout(
                shapes=shapes,
                annotations=annotations,
                height=400
            )
            
            st.plotly_chart(fig_rolling_var, use_container_width=True)

        return rolling_variance, rolling_loadings_pc1, rolling_dates

    except Exception as e:
        st.error("Erro ao executar análise PCA rolling: " + str(e))
        return [], [], []


def display_rolling_pca_stability(
    rolling_variance, rolling_loadings_pc1, rolling_dates, selected_assets
):
    """Exibe análise de estabilidade do PCA rolling"""
    # Estabilidade dos loadings
    loadings_df = pd.DataFrame(
        rolling_loadings_pc1, columns=selected_assets, index=rolling_dates
    )

    # Plot dos loadings principais ao longo do tempo
    main_assets = loadings_df.abs().mean().nlargest(4).index
    fig_loadings_time = go.Figure()

    for asset in main_assets:
        fig_loadings_time.add_trace(
            go.Scatter(
                x=loadings_df.index, y=loadings_df[asset], name=asset, mode="lines"
            )
        )

    fig_loadings_time.update_layout(
        title="🎯 Evolução dos Loadings Principais (PC1)",
        xaxis_title="Data",
        yaxis_title="Loading",
        height=400,
    )
    st.plotly_chart(fig_loadings_time, use_container_width=True)

    # Métricas de estabilidade
    col1, col2, col3 = st.columns(3)
    with col1:
        var_stability = np.std(rolling_variance)
        st.metric(
            "Estabilidade Variância",
            f"{var_stability:.4f}",
            help="Menor = mais estável",
        )
    with col2:
        mean_var = np.mean(rolling_variance)
        st.metric("Variância Média PC1", f"{mean_var:.2%}")
    with col3:
        loading_stability = loadings_df.std().mean()
        st.metric(
            "Estabilidade Loadings",
            f"{loading_stability:.4f}",
            help="Menor = mais estável",
        )


def build_pca_portfolio(
    pca, components, returns_selected, selected_assets, strategy_type, rebalance_freq
):
    """Constrói portfolio baseado em PCA"""
    st.subheader("🎯 Construção de Portfolio via PCA")
    st.info(
        "💡 Use os componentes principais para construir portfolios otimizados (Rebalanceamento: "
        + str(rebalance_freq)
        + ")"
    )
    
    # Filtrar apenas os ativos selecionados
    if selected_assets is not None and len(selected_assets) > 0:
        returns_selected = returns_selected[selected_assets]

    # Estratégias baseadas em PCA
    if st.button("🚀 Construir Portfolio PCA"):
        with st.spinner("Construindo portfolio..."):
            # Usar componentes para construir pesos
            if strategy_type == "Maximum Diversification":
                # Pesos baseados no primeiro componente (invertidos para diversificação)
                weights_raw = 1 / (abs(pca.components_[0]) + 0.001)
                weights = weights_raw / weights_raw.sum()
            elif strategy_type == "Minimum Variance":
                # Pesos baseados na matriz de covariância dos componentes
                cov_components = np.cov(components[:, :3].T)
                inv_cov = np.linalg.pinv(cov_components)
                weights_components = (
                    inv_cov
                    @ np.ones(len(inv_cov))
                    / (np.ones(len(inv_cov)) @ inv_cov @ np.ones(len(inv_cov)))
                )
                weights = abs(pca.components_[:3].T @ weights_components)
                weights = weights / weights.sum()
            else:  # Equal Risk Contribution
                # Pesos que equalizam a contribuição de risco
                vol = returns_selected.std()
                weights = (1 / vol) / (1 / vol).sum()

            # Construir portfolio
            portfolio_returns = (returns_selected * weights).sum(axis=1)
            portfolio_cumulative = (1 + portfolio_returns).cumprod()

            # Calcular métricas
            total_return = portfolio_cumulative.iloc[-1] - 1
            annual_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
            annual_vol = portfolio_returns.std() * np.sqrt(252)
            sharpe = annual_return / annual_vol if annual_vol > 0 else 0

            return (
                weights,
                portfolio_cumulative,
                total_return,
                annual_return,
                annual_vol,
                sharpe,
            )

    return None, None, None, None, None, None


def display_portfolio_results(
    weights,
    portfolio_cumulative,
    total_return,
    annual_return,
    annual_vol,
    sharpe,
    selected_assets,
    strategy_type,
):
    """Exibe resultados do portfolio PCA"""
    if weights is not None:
        # Calcular retorno mensal a partir do retorno anual
        monthly_return = (1 + annual_return) ** (1/12) - 1
        
        # Exibir resultados
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric(TOTAL_RETURN_LABEL, format_percentage(total_return))
        with col2:
            st.metric("Retorno Anual", format_percentage(annual_return))
        with col3:
            st.metric("Retorno Mensal", format_percentage(monthly_return))
        with col4:
            st.metric("Volatilidade", format_percentage(annual_vol))
        with col5:
            st.metric("Sharpe Ratio", format_ratio(sharpe))

        # Gráfico de alocação
        weights_df = pd.DataFrame(
            {"Ativo": selected_assets, WEIGHT_PERCENTAGE_LABEL: weights * 100}
        ).sort_values(WEIGHT_PERCENTAGE_LABEL, ascending=False)

        fig_allocation = px.pie(
            weights_df,
            values=WEIGHT_PERCENTAGE_LABEL,
            names="Ativo",
            title="🎯 Alocação do Portfolio (" + str(strategy_type) + ")",
        )
        st.plotly_chart(fig_allocation, use_container_width=True)

        # Performance histórica
        fig_performance = px.line(
            x=portfolio_cumulative.index,
            y=(portfolio_cumulative - 1) * 100,
            title="📈 Performance do Portfolio PCA",
            labels={"x": "Data", "y": "Retorno Acumulado (%)"},
        )
        st.plotly_chart(fig_performance, use_container_width=True)


def analyze_pca_risk(pca, selected_assets, n_components):
    """Analisa risco via PCA"""
    st.subheader("📉 Análise de Risco via PCA")
    st.info("💡 Use PCA para identificar os principais fatores de risco do portfolio")

    # Decomposição de risco por componente
    risk_contributions = []
    for i in range(min(5, n_components)):
        # Variância explicada por cada componente
        component_var = pca.explained_variance_[i]
        # Contribuição dos ativos para este componente
        loadings = abs(pca.components_[i])
        # Risco explicado por componente
        risk_contrib = component_var * loadings
        risk_contributions.append(risk_contrib)

    # Matriz de contribuições de risco
    risk_matrix = pd.DataFrame(
        risk_contributions,
        columns=selected_assets,
        index=[f"PC{i+1}" for i in range(len(risk_contributions))],
    )

    # Heatmap de risco
    fig_risk = px.imshow(
        risk_matrix.to_numpy(),
        x=risk_matrix.columns,
        y=risk_matrix.index,
        color_continuous_scale="Reds",
        title="🔥 Mapa de Calor - Contribuição de Risco por Componente",
    )
    st.plotly_chart(fig_risk, use_container_width=True)
    
    # Top riscos
    total_risk_by_asset = risk_matrix.sum()
    top_risks = total_risk_by_asset.nlargest(5)
    st.subheader("⚠️ Principais Fatores de Risco")
    for i, (asset, risk) in enumerate(top_risks.items()):
        st.write(f"{i+1}. **{asset}**: Contribuição de risco = {risk:.4f}")

    # Diversificação via PCA
    effective_components = (pca.explained_variance_ratio_**2).sum() ** -1
    st.metric(
        "Número Efetivo de Componentes",
        f"{effective_components:.1f}",
        help="Maior = mais diversificado",
    )


def display_interactive_pca_example(pca, selected_assets, n_components):
    """Exibe exemplo interativo de PCA"""
    st.subheader("🔧 Exemplo Interativo")

    example_component = st.selectbox(
        "Selecione um componente para análise detalhada:",
        [f"PC{i+1}" for i in range(min(3, n_components))],
    )

    pc_idx = int(example_component[2:]) - 1

    # Análise detalhada do componente selecionado
    st.write(
        f"**{example_component}** explica {pca.explained_variance_ratio_[pc_idx]:.1%} da variância total"
    )

    # Top contributors
    component_loadings = pd.DataFrame(
        {
            "Ativo": selected_assets,
            "Loading": pca.components_[pc_idx],
            "Abs_Loading": abs(pca.components_[pc_idx]),
        }
    ).sort_values("Abs_Loading", ascending=False)

    st.write("**Principais contribuidores:**")
    for i, row in component_loadings.head(3).iterrows():
        direction = "positivamente" if row["Loading"] > 0 else "negativamente"
        st.write(f"- **{row['Ativo']}**: Contribui {direction} ({row['Loading']:.3f})")


def display_pca_educational_content():
    """Exibe conteúdo educacional sobre PCA"""
    st.subheader("🔬 Explicação Didática - PCA em Finanças")

    st.markdown(
        """
    ### 📚 Conceitos Fundamentais
    
    **1. O que é PCA?**
    - Análise de Componentes Principais reduz dados multidimensionais
    - Encontra direções de máxima variância nos dados
    - Cada componente é uma combinação linear dos ativos originais
    
    **2. Como interpretar os resultados?**
    
    #### 🎯 Variância Explicada
    - **PC1 com 60%**: O primeiro fator explica 60% da variação total do mercado
    - **PC1+PC2 com 80%**: Os dois primeiros fatores capturam 80% dos movimentos
    
    #### 📊 Loadings (Cargas)
    - **Loading positivo alto**: Ativo se move na mesma direção do componente
    - **Loading negativo alto**: Ativo se move na direção oposta
    - **Loading próximo de zero**: Ativo não está relacionado a este fator
    
    **3. Aplicações Práticas em Portfolios**
    
    #### 🎯 Construção de Portfolios
    - **Maximum Diversification**: Minimiza exposição ao fator dominante
    - **Factor Investing**: Investe especificamente em fatores identificados
    - **Risk Budgeting**: Aloca risco entre diferentes componentes
    
    #### 📉 Gestão de Risco
    - **Identificação de fatores**: Quais são os principais drivers de risco?
    - **Stress testing**: Como o portfolio reage a choques nos componentes?
    - **Hedging**: Construir hedges baseados nos fatores identificados
    
    **4. Limitações e Cuidados**
    
    ⚠️ **Estabilidade temporal**: Componentes podem mudar ao longo do tempo
    
    ⚠️ **Interpretação econômica**: Nem sempre os componentes têm significado econômico claro
    
    ⚠️ **Outliers**: Dados extremos podem distorcer os componentes
    
    ### 💡 Dicas Práticas
    
    - Use janelas rolling para capturar mudanças de regime
    - Combine PCA com conhecimento econômico/setorial
    - Monitore a estabilidade dos loadings
    - Considere re-estimar periodicamente
    """
    )
