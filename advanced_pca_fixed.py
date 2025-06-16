from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import plotly.express as px
import streamlit as st

def execute_rolling_pca_analysis(returns_selected, window_size, rebalance_freq, rebalance_window):
    """Executa análise PCA rolling"""
    st.subheader("📈 Análise PCA Rolling")
    st.info("💡 Analisa como os componentes principais evoluem ao longo do tempo")
    
    try:
        # Verificar se window_size é um número inteiro válido
        window_size_int = int(window_size)
        rebalance_window_int = int(rebalance_window)
        
        # Calcular PCA rolling
        rolling_variance = []
        rolling_loadings_pc1 = []
        rolling_dates = []
        
        with st.spinner("Calculando PCA rolling..."):
            for i in range(window_size_int, len(returns_selected)):
                window_data = returns_selected.iloc[i-window_size_int:i]
                
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
            labels={'x': 'Data', 'y': 'Variância Explicada PC1'}
        )
        
        # Adicionar marcadores de rebalanceamento
        rebalance_dates = []
        for i in range(0, len(rolling_dates), rebalance_window_int):
            if i < len(rolling_dates):
                rebalance_dates.append(rolling_dates[i])
        
        # Texto para anotação
        annotation_text = "Rebalanceamento " + str(rebalance_freq)
        
        # Adicionar linhas verticais para datas de rebalanceamento
        for rebalance_date in rebalance_dates:
            try:
                # Converter para timestamp se necessário
                if not isinstance(rebalance_date, pd.Timestamp):
                    rebalance_date = pd.Timestamp(rebalance_date)
                
                # Converter para string no formato YYYY-MM-DD
                date_str = rebalance_date.strftime('%Y-%m-%d')
                
                # Adicionar linha vertical
                fig_rolling_var.add_vline(
                    x=date_str, 
                    line_dash="dot",
                    line_color="red",
                    annotation_text=annotation_text
                )
            except Exception as e:
                st.warning("Erro ao adicionar linha de rebalanceamento: " + str(e))
        
        fig_rolling_var.update_layout(height=400)
        st.plotly_chart(fig_rolling_var, use_container_width=True)
        
        return rolling_variance, rolling_loadings_pc1, rolling_dates
    
    except Exception as e:
        st.error("Erro ao executar análise PCA rolling: " + str(e))
        return [], [], []
