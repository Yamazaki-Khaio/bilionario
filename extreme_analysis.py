#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo para análise detalhada de eventos extremos para qualquer ativo
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy import stats
from scipy.stats import skew, kurtosis
from scipy.optimize import minimize

def execute_extreme_analysis(stat_analyzer, selected_asset, threshold):
    """
    Executa análise de eventos extremos para qualquer ativo selecionado.
    
    Parameters:
    -----------
    stat_analyzer : StatisticalAnalysis
        Objeto de análise estatística
    selected_asset : str
        Nome do ativo selecionado
    threshold : float
        Threshold para considerar evento extremo (ex: 0.10 para 10%)
        
    Returns:
    --------
    None
        Exibe os resultados diretamente via streamlit
    """
    with st.spinner(f"Analisando extremos do ativo {selected_asset}..."):
        try:
            # Verificar se o ativo é válido
            if selected_asset not in stat_analyzer.returns.columns:
                st.error(f"❌ Ativo {selected_asset} não encontrado nos dados")
                return
            
            # Realizar análise detalhada usando a função extreme_analysis_any_asset
            if hasattr(stat_analyzer, 'extreme_analysis_any_asset'):
                extreme_analysis = stat_analyzer.extreme_analysis_any_asset(selected_asset, threshold)
            else:
                # Implementação alternativa caso o método não exista
                extreme_analysis = _fallback_extreme_analysis(stat_analyzer, selected_asset, threshold)
            
            if 'error' in extreme_analysis:
                st.error(f"❌ {extreme_analysis['error']}")
                return
            
            # Obter métricas principais
            extreme_count = extreme_analysis.get('extreme_falls_count', 0)
            total_days = extreme_analysis.get('total_days', 0)
            prob_empirical = extreme_analysis.get('probability', 0)
            extreme_dates = extreme_analysis.get('extreme_falls_dates', [])
            extreme_values = extreme_analysis.get('extreme_falls_values', [])
            
            # Exibir cabeçalho
            st.subheader(f"📊 Análise de Quedas > {threshold:.0%} - {selected_asset}")
            
            # Verificar se possui dados suficientes
            if total_days < 30:
                st.warning(f"⚠️ Dados insuficientes ({total_days} observações) para análise robusta.")
            
            # Mostrar métricas principais
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total de Dias", f"{total_days}")
                
            with col2:
                st.metric("Quedas Extremas", f"{extreme_count}")
                
            with col3:
                st.metric("Probabilidade Empírica", f"{prob_empirical:.2%}")
            
            # Obter dados de retornos para visualização
            asset_returns = stat_analyzer.returns[selected_asset].pct_change().dropna()
            
            # Análise detalhada com tabs
            st.write("### 🔍 Análise Detalhada de Eventos Extremos")
            
            # Criar tabs para diferentes visualizações
            tab_empirical, tab_normal, tab_tstudent, tab_historico = st.tabs([
                "📊 Empírica", 
                "🔄 Normal", 
                "📈 t-Student", 
                "🗓️ Histórico"
            ])
            
            with tab_empirical:
                _show_empirical_analysis(asset_returns, threshold, prob_empirical, extreme_count)
                
            with tab_normal:
                _show_normal_distribution_analysis(asset_returns, threshold, prob_empirical)
                
            with tab_tstudent:
                _show_tstudent_distribution_analysis(asset_returns, threshold, prob_empirical, selected_asset)
                
            with tab_historico:
                _show_historical_extremes(asset_returns, extreme_dates, extreme_values, selected_asset, threshold)
                
            # Interpretação geral dos resultados
            _show_general_interpretation(asset_returns, threshold, prob_empirical, selected_asset)
            
        except Exception as e:
            st.error(f"Erro na análise: {str(e)}")
            import traceback
            st.error(f"Detalhe: {traceback.format_exc()}")

def _fallback_extreme_analysis(stat_analyzer, selected_asset, threshold):
    """Implementação alternativa de análise extrema caso o método não exista"""
    asset_returns = stat_analyzer.returns[selected_asset].pct_change().dropna()
    extreme_falls = asset_returns[asset_returns <= -threshold]
    
    return {
        'asset_symbol': selected_asset,
        'threshold': threshold,
        'total_days': len(asset_returns),
        'extreme_falls_count': len(extreme_falls),
        'probability': len(extreme_falls) / len(asset_returns) if len(asset_returns) > 0 else 0,
        'extreme_falls_dates': extreme_falls.index.tolist(),
        'extreme_falls_values': extreme_falls.values.tolist(),
        'daily_statistics': {
            'mean': asset_returns.mean(),
            'std': asset_returns.std(),
            'skewness': skew(asset_returns) if len(asset_returns) > 3 else 0,
            'kurtosis': kurtosis(asset_returns, fisher=True) if len(asset_returns) > 3 else 0,
            'min': asset_returns.min(),
            'max': asset_returns.max()
        }
    }

def _show_empirical_analysis(asset_returns, threshold, prob_empirical, extreme_count):
    """Exibe análise empírica de eventos extremos"""
    st.markdown("##### Análise Empírica")
    
    # Criar histograma com densidade
    fig = go.Figure()
    
    # Adicionar histograma com densidade de probabilidade
    fig.add_trace(go.Histogram(
        x=asset_returns,
        histnorm='probability density',
        name="Retornos",
        opacity=0.6,
        marker_color='#1f77b4'
    ))
    
    # Adicionar linha vertical para o threshold
    fig.add_vline(
        x=-threshold, 
        line_dash="dash", 
        line_color="red",
        annotation_text=f"Threshold: -{threshold:.0%}",
        annotation_position="top right"
    )
    
    # Customizar layout
    fig.update_layout(
        title="Distribuição Empírica de Retornos",
        xaxis_title="Retorno Diário",
        yaxis_title="Densidade de Probabilidade",
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Mostrar probabilidade empirica com mais detalhes
    if extreme_count > 0:
        st.info(f"""
        📊 **Probabilidade Empírica**: {prob_empirical:.2%}
        
        Com base nos dados históricos, a probabilidade de uma queda diária superior a {threshold:.0%} é de {prob_empirical:.2%}.
        Isso equivale a aproximadamente 1 queda a cada {1/prob_empirical:.0f} dias de negociação, ou cerca de {252/prob_empirical:.1f} dias úteis por ano.
        """)
    else:
        st.info("Não foram observadas quedas superiores ao threshold no período analisado.")

def _show_normal_distribution_analysis(asset_returns, threshold, prob_empirical):
    """Exibe análise usando distribuição normal"""
    st.markdown("##### Modelagem com Distribuição Normal")
    
    # Parâmetros da distribuição Normal
    mu = asset_returns.mean()
    sigma = asset_returns.std()
    
    # Probabilidade teórica baseada na Normal
    prob_normal = stats.norm.cdf(-threshold, mu, sigma)
    
    # Criar gráfico
    x = np.linspace(asset_returns.min(), asset_returns.max(), 1000)
    y = stats.norm.pdf(x, mu, sigma)
    
    fig = go.Figure()
    
    # Adicionar histograma com densidade de probabilidade
    fig.add_trace(go.Histogram(
        x=asset_returns,
        histnorm='probability density',
        name="Retornos",
        opacity=0.6,
        marker_color='#1f77b4'
    ))
    
    # Adicionar curva de distribuição normal
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='lines',
        name='Normal',
        line=dict(color='red', width=2)
    ))
    
    # Área sombreada para quedas extremas
    x_extreme = np.linspace(asset_returns.min(), -threshold, 100)
    y_extreme = stats.norm.pdf(x_extreme, mu, sigma)
    
    fig.add_trace(go.Scatter(
        x=x_extreme,
        y=y_extreme,
        fill='tozeroy',
        fillcolor='rgba(255,0,0,0.2)',
        line=dict(color='rgba(255,0,0,0)'),
        name=f'Prob. Normal: {prob_normal:.2%}'
    ))
    
    # Adicionar linha vertical para o threshold
    fig.add_vline(
        x=-threshold, 
        line_dash="dash", 
        line_color="red",
        annotation_text=f"Threshold: -{threshold:.0%}",
        annotation_position="top right"
    )
    
    # Customizar layout
    fig.update_layout(
        title="Modelagem com Distribuição Normal",
        xaxis_title="Retorno Diário",
        yaxis_title="Densidade de Probabilidade",
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Mostrar comparação entre probabilidade empírica e teórica
    ratio = prob_empirical / prob_normal if prob_normal > 0 else 0
    
    if ratio > 1.3:
        st.warning(f"""
        ⚠️ **Alerta**: A probabilidade empírica ({prob_empirical:.2%}) é {ratio:.1f}x maior que a estimada pela distribuição Normal ({prob_normal:.2%}).
        
        Isso indica que o ativo possui **caudas mais pesadas** do que o previsto pela Normal, subestimando o risco de eventos extremos.
        """)
    elif ratio < 0.7 and ratio > 0:
        st.info(f"""
        ℹ️ **Observação**: A probabilidade empírica ({prob_empirical:.2%}) é {1/ratio:.1f}x menor que a estimada pela distribuição Normal ({prob_normal:.2%}).
        
        Isso pode indicar que o período analisado teve menos eventos extremos do que o esperado teoricamente.
        """)
    else:
        st.success(f"""
        ✅ **Validação**: A probabilidade empírica ({prob_empirical:.2%}) é relativamente próxima da estimada pela distribuição Normal ({prob_normal:.2%}).
        
        A modelagem Normal captura razoavelmente bem o comportamento de quedas do ativo neste threshold.
        """)

def _show_tstudent_distribution_analysis(asset_returns, threshold, prob_empirical, asset_symbol=None):
    """Exibe análise usando distribuição t-Student"""
    st.markdown("##### Modelagem com Distribuição t-Student")
    
    # Parâmetros da distribuição Normal
    mu = asset_returns.mean()
    sigma = asset_returns.std()
    
    # Estimar parâmetros da t-Student (graus de liberdade)
    def t_loglikelihood(params, data):
        df, loc, scale = params
        return -np.sum(stats.t.logpdf(data, df=df, loc=loc, scale=scale))
    
    # Estimativa inicial baseada em momentos
    initial_params = [6, mu, sigma]
    
    try:
        # Usar otimização para encontrar melhores parâmetros
        result = minimize(t_loglikelihood, initial_params, args=(asset_returns,), 
                         bounds=[(2.1, 50), (None, None), (0.0001, None)])
        
        df_param, loc_param, scale_param = result.x
        
        # Probabilidade teórica baseada na t-Student
        prob_t = stats.t.cdf(-threshold, df=df_param, loc=loc_param, scale=scale_param)
        
        # Criar gráfico
        x = np.linspace(asset_returns.min(), asset_returns.max(), 1000)
        y_t = stats.t.pdf(x, df=df_param, loc=loc_param, scale=scale_param)
        y_norm = stats.norm.pdf(x, mu, sigma)
        
        fig = go.Figure()
        
        # Adicionar histograma com densidade de probabilidade
        fig.add_trace(go.Histogram(
            x=asset_returns,
            histnorm='probability density',
            name="Retornos",
            opacity=0.4,
            marker_color='#1f77b4'
        ))
        
        # Adicionar curva de distribuição t-Student
        fig.add_trace(go.Scatter(
            x=x,
            y=y_t,
            mode='lines',
            name='t-Student',
            line=dict(color='red', width=2)
        ))
        
        # Adicionar curva de distribuição normal para comparação
        fig.add_trace(go.Scatter(
            x=x,
            y=y_norm,
            mode='lines',
            name='Normal',
            line=dict(color='green', width=2, dash='dash')
        ))
        
        # Área sombreada para quedas extremas (t-Student)
        x_extreme = np.linspace(asset_returns.min(), -threshold, 100)
        y_extreme = stats.t.pdf(x_extreme, df=df_param, loc=loc_param, scale=scale_param)
        
        fig.add_trace(go.Scatter(
            x=x_extreme,
            y=y_extreme,
            fill='tozeroy',
            fillcolor='rgba(255,0,0,0.2)',
            line=dict(color='rgba(255,0,0,0)'),
            name=f'Prob. t-Student: {prob_t:.2%}'
        ))
        
        # Adicionar linha vertical para o threshold
        fig.add_vline(
            x=-threshold, 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"Threshold: -{threshold:.0%}",
            annotation_position="top right"
        )
        
        # Customizar layout
        fig.update_layout(
            title=f"Modelagem com Distribuição t-Student (v={df_param:.1f})",
            xaxis_title="Retorno Diário",
            yaxis_title="Densidade de Probabilidade",
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Mostrar comparação entre probabilidades        ratio_t = prob_empirical / prob_t if prob_t > 0 else 0
        
        # Cálculo da probabilidade normal
        prob_normal = stats.norm.cdf(-threshold, mu, sigma)
        
        # Tratamento para evitar nan%
        prob_normal_display = "0.00%" if np.isnan(prob_normal) else f"{prob_normal:.2%}"
        prob_t_display = "0.00%" if np.isnan(prob_t) else f"{prob_t:.2%}"
        
        normal_ratio = "0.00" if np.isnan(prob_normal) or prob_empirical == 0 else f"{prob_normal/prob_empirical:.2f}"
        t_ratio = "0.00" if np.isnan(prob_t) or prob_empirical == 0 else f"{prob_t/prob_empirical:.2f}"        # Tabela comparativa
        comp_df = pd.DataFrame({
            "Modelo": ["Empírico", "Normal", "t-Student"],
            "Probabilidade": [f"{prob_empirical:.2%}", prob_normal_display, prob_t_display],
            "Razão p/ Empírico": ["1.00", normal_ratio, t_ratio]
        })
        st.table(comp_df)
        
        # Adicionar botão de download para esta análise específica
        from pdf_export_helpers import add_download_buttons_to_extreme_analysis
        add_download_buttons_to_extreme_analysis(
            selected_asset=asset_symbol, 
            threshold=threshold, 
            prob_empirical=prob_empirical, 
            prob_normal=prob_normal, 
            prob_t=prob_t, 
            df_param=df_param
        )
        
        # Adicionar botão de download para esta análise específica
        try:
            from pdf_export_helpers import add_download_buttons_to_extreme_analysis
            add_download_buttons_to_extreme_analysis(
                asset_symbol, threshold, prob_empirical, prob_normal, prob_t, df_param
            )
        except Exception as e:
            st.warning(f"Não foi possível adicionar opção de download: {str(e)}")
        
        if abs(ratio_t - 1) < 0.2 and ratio_t > 0:
            st.success(f"""
            ✅ **Validação**: A distribuição t-Student com {df_param:.1f} graus de liberdade modela bem os eventos extremos deste ativo.
            
            A probabilidade estimada pela t-Student ({prob_t:.2%}) está muito próxima da probabilidade empírica ({prob_empirical:.2%}).
            """)
        elif ratio_t > 1:
            st.warning(f"""
            ⚠️ **Alerta**: A probabilidade empírica ({prob_empirical:.2%}) ainda é maior que a estimada pela t-Student ({prob_t:.2%}).
            
            Isso sugere que mesmo a modelagem com t-Student pode estar subestimando o risco de quedas extremas neste ativo.
            """)
        else:
            st.info(f"""
            ℹ️ **Observação**: A modelagem com t-Student ({prob_t:.2%}) fornece uma estimativa mais conservadora que a probabilidade empírica ({prob_empirical:.2%}).
            
            Isso pode ser adequado para modelagem de risco com margem de segurança.
            """)
            
    except Exception as e:
        st.warning(f"Não foi possível estimar os parâmetros da distribuição t-Student: {str(e)}")
        st.info("Verifique se há dados suficientes ou tente novamente com um conjunto de dados maior.")

def _show_historical_extremes(asset_returns, extreme_dates, extreme_values, asset_symbol, threshold):
    """Exibe histórico de eventos extremos"""
    st.markdown("##### Datas de Quedas Extremas")
    
    if extreme_dates and len(extreme_dates) > 0:
        # Converter timestamps para strings formatadas
        if isinstance(extreme_dates[0], pd.Timestamp):
            date_strings = [date.strftime('%d/%m/%Y') for date in extreme_dates]
        else:
            date_strings = extreme_dates
        
        # Obter os retornos para essas datas
        extreme_returns_values = []
        for date in extreme_dates:
            try:
                if isinstance(date, str):
                    date = pd.to_datetime(date)
                if date in asset_returns.index:
                    extreme_returns_values.append(asset_returns[date])
                else:
                    extreme_returns_values.append(None)
            except:
                extreme_returns_values.append(None)
        
        # Usar os valores diretos se possível
        if len(extreme_values) == len(extreme_dates) and all(v is not None for v in extreme_values):
            display_values = extreme_values
        else:
            display_values = extreme_returns_values
        
        # Criar dataframe para exibição
        extreme_df = pd.DataFrame({
            "Data": date_strings,
            "Queda (%)": [f"{ret*100:.2f}%" if ret is not None else "N/A" for ret in display_values]
        })
        
        st.dataframe(extreme_df, use_container_width=True)
        
        # Oferecer botão para baixar dados de quedas extremas
        csv = extreme_df.to_csv(index=False)
        st.download_button(
            label="📥 Baixar Dados de Quedas Extremas",
            data=csv,
            file_name=f'quedas_extremas_{asset_symbol}.csv',
            mime='text/csv',
        )
    else:
        st.info("Não foram observadas quedas superiores ao threshold no período analisado.")

def _show_general_interpretation(asset_returns, threshold, prob_empirical, asset_symbol):
    """Exibe interpretação geral dos resultados"""
    st.markdown("#### 💡 Interpretação dos Resultados")
    
    # Parâmetros da distribuição
    mu = asset_returns.mean()
    sigma = asset_returns.std()
    
    # Probabilidades dos modelos
    prob_normal = stats.norm.cdf(-threshold, mu, sigma)
    
    # Determinar o modelo mais adequado para esta análise
    # Inicialmente usamos o empírico, mas tentamos determinar o melhor modelo
    best_model = "empírico"
    best_prob = prob_empirical
    
    # Verificar razão entre probabilidade empírica e normal
    ratio_normal = prob_empirical / prob_normal if prob_normal > 0 and prob_empirical > 0 else 0
    
    # Tentar obter resultados da t-student se possível
    try:
        def t_loglikelihood(params, data):
            df, loc, scale = params
            return -np.sum(stats.t.logpdf(data, df=df, loc=loc, scale=scale))
        
        initial_params = [6, mu, sigma]
        result = minimize(t_loglikelihood, initial_params, args=(asset_returns,), 
                         bounds=[(2.1, 50), (None, None), (0.0001, None)])
        
        df_param, loc_param, scale_param = result.x
        prob_t = stats.t.cdf(-threshold, df=df_param, loc=loc_param, scale=scale_param)
        ratio_t = prob_empirical / prob_t if prob_t > 0 and prob_empirical > 0 else 0
        
        # Verificar qual modelo melhor se ajusta aos dados empíricos
        if abs(ratio_t - 1) < 0.2 and prob_t > 0:
            best_model = "t-Student"
            best_prob = prob_t
        elif abs(ratio_normal - 1) < 0.2 and prob_normal > 0:
            best_model = "Normal"
            best_prob = prob_normal
    except:
        # Em caso de falha, continuamos com o modelo empírico
        pass
        
    # Interpretar em termos práticos
    if best_prob > 0.05:  # 5%
        st.warning(f"""
        ⚠️ **Alto Risco**: Baseado no modelo {best_model}, a probabilidade de {best_prob:.2%} para quedas diárias superiores a {threshold:.0%} 
        para o ativo {asset_symbol} indica volatilidade elevada.
        
        **Recomendações:**
        - Considere estratégias de hedge (opções de venda, stop-loss)
        - Diversifique o portfólio para reduzir exposição
        - Monitore atentamente fatores externos que podem amplificar quedas
        """)
    elif best_prob > 0.02:  # 2%
        st.info(f"""
        💡 **Risco Moderado**: Baseado no modelo {best_model}, a probabilidade de {best_prob:.2%} para quedas diárias superiores a {threshold:.0%}
        para o ativo {asset_symbol} é significativa.
        
        **Recomendações:**
        - Monitore indicadores macro e setoriais que podem afetar o ativo
        - Mantenha um plano de contingência para eventos negativos
        - Considere um mix de posições de longo prazo e proteções táticas
        """)
    else:
        st.success(f"""
        ✅ **Risco Controlado**: Baseado no modelo {best_model}, a probabilidade de {best_prob:.2%} para quedas diárias superiores a {threshold:.0%}
        para o ativo {asset_symbol} é relativamente baixa no horizonte analisado.
        
        **Recomendações:**
        - Mantenha monitoramento regular dos indicadores de risco
        - Reavalie periodicamente essa análise, especialmente após mudanças de mercado significativas
        - Considere este ativo para estratégias de longo prazo com uma tolerância controlada ao risco
        """)
