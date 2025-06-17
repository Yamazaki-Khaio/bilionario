#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo para auxiliar na exportação de análises para PDF
Funções de suporte para exportação de relatórios em formato PDF
"""
import streamlit as st
import base64
from io import BytesIO
import pandas as pd
from datetime import datetime
from scipy.stats import skew, kurtosis
import numpy as np

from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch

def create_download_button(pdf_data, filename="relatório.pdf", button_text="Baixar PDF"):
    """Cria um botão de download para um PDF gerado"""
    b64_pdf = base64.b64encode(pdf_data).decode()
    
    # Criar botão de download estilizado
    download_button_html = f'''
    <a href="data:application/pdf;base64,{b64_pdf}" download="{filename}">
        <div style="
            display: inline-flex;
            align-items: center;
            background-color: #4CAF50;
            color: white;
            padding: 12px 20px;
            border-radius: 4px;
            cursor: pointer;
            text-decoration: none;
            font-weight: bold;
            margin: 10px 0;">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" fill="currentColor" class="bi bi-download" viewBox="0 0 16 16" style="margin-right: 8px;">
                <path d="M.5 9.9a.5.5 0 0 1 .5.5v2.5a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-2.5a.5.5 0 0 1 1 0v2.5a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2v-2.5a.5.5 0 0 1 .5-.5z"/>
                <path d="M7.646 11.854a.5.5 0 0 0 .708 0l3-3a.5.5 0 0 0-.708-.708L8.5 10.293V1.5a.5.5 0 0 0-1 0v8.793L5.354 8.146a.5.5 0 1 0-.708.708l3 3z"/>
            </svg>
            {button_text}
        </div>
    </a>
    '''
    
    return download_button_html

def generate_extreme_analysis_pdf(asset_symbol, threshold, prob_empirical, prob_normal, prob_t, df_param):
    """Gera PDF para análise de eventos extremos"""
    import matplotlib.pyplot as plt
    import io
    from reportlab.lib.utils import ImageReader
    
    buffer = BytesIO()
    pdf = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []
    
    # Título e cabeçalho
    title_style = ParagraphStyle(
        name="Title",
        fontSize=18,
        alignment=1,
        spaceAfter=12
    )
    
    elements.append(Paragraph(f"Análise de Eventos Extremos - {asset_symbol}", title_style))
    elements.append(Spacer(1, 0.2*inch))
    elements.append(Paragraph(f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}", styles['Normal']))
    elements.append(Spacer(1, 0.3*inch))
    
    # Parâmetros da análise
    elements.append(Paragraph("Parâmetros de Análise:", styles['Heading2']))
    elements.append(Spacer(1, 0.1*inch))
    elements.append(Paragraph(f"• Ativo analisado: {asset_symbol}", styles['Normal']))
    elements.append(Paragraph(f"• Threshold para eventos extremos: {threshold:.2%}", styles['Normal']))
    elements.append(Spacer(1, 0.2*inch))
    
    # Resultados de probabilidade
    elements.append(Paragraph("Comparação de Modelos:", styles['Heading2']))
    elements.append(Spacer(1, 0.1*inch))
    
    # Tabela de comparação
    prob_normal_display = "0.00%" if np.isnan(prob_normal) else f"{prob_normal:.2%}"
    prob_t_display = "0.00%" if np.isnan(prob_t) else f"{prob_t:.2%}"
    
    normal_ratio = "0.00" if np.isnan(prob_normal) or prob_empirical == 0 else f"{prob_normal/prob_empirical:.2f}"
    t_ratio = "0.00" if np.isnan(prob_t) or prob_empirical == 0 else f"{prob_t/prob_empirical:.2f}"
    
    data = [
        ["Modelo", "Probabilidade", "Razão p/ Empírico"],
        ["Empírico", f"{prob_empirical:.2%}", "1.00"],
        ["Normal", prob_normal_display, normal_ratio],
        ["t-Student", prob_t_display, t_ratio]
    ]
    
    table = Table(data, colWidths=[2*inch, 1.5*inch, 1.5*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    
    elements.append(table)
    elements.append(Spacer(1, 0.2*inch))
    
    # Interpretação
    elements.append(Paragraph("Interpretação dos Resultados:", styles['Heading2']))
    elements.append(Spacer(1, 0.1*inch))
    
    # Adicionar interpretação adequada
    ratio_t = prob_empirical / prob_t if prob_t > 0 and not np.isnan(prob_t) else 0
    
    if abs(ratio_t - 1) < 0.2 and ratio_t > 0:
        elements.append(Paragraph(f"✅ A distribuição t-Student com {df_param:.1f} graus de liberdade modela bem os eventos extremos deste ativo.", 
                                 styles['Normal']))
        elements.append(Paragraph(f"A probabilidade estimada pela t-Student ({prob_t:.2%}) está muito próxima da probabilidade empírica ({prob_empirical:.2%}).", 
                                 styles['Normal']))
    elif ratio_t > 1:
        elements.append(Paragraph(f"⚠️ A probabilidade empírica ({prob_empirical:.2%}) ainda é maior que a estimada pela t-Student ({prob_t:.2%}).", 
                                 styles['Normal']))
        elements.append(Paragraph("Isso sugere que mesmo a modelagem com t-Student pode estar subestimando o risco de quedas extremas neste ativo.", 
                                 styles['Normal']))
    else:
        elements.append(Paragraph(f"ℹ️ A modelagem com t-Student ({prob_t:.2%}) fornece uma estimativa mais conservadora que a probabilidade empírica ({prob_empirical:.2%}).", 
                                 styles['Normal']))
        elements.append(Paragraph("Isso pode ser adequado para modelagem de risco com margem de segurança.", 
                                 styles['Normal']))
    
    # Recomendações
    elements.append(Spacer(1, 0.3*inch))
    elements.append(Paragraph("Recomendações:", styles['Heading2']))
    elements.append(Spacer(1, 0.1*inch))
    
    # Determinar melhor modelo
    best_model = "empírico"
    best_prob = prob_empirical
    
    if prob_t > 0 and not np.isnan(prob_t) and abs(ratio_t - 1) < 0.2:
        best_model = "t-Student"
        best_prob = prob_t
    elif prob_normal > 0 and not np.isnan(prob_normal) and abs(prob_empirical/prob_normal - 1) < 0.2:
        best_model = "Normal"
        best_prob = prob_normal
    
    elements.append(Paragraph(f"• Recomendamos utilizar o modelo {best_model} para estimativas de risco.", styles['Normal']))
    
    if best_prob > 0.05:
        elements.append(Paragraph(f"• O ativo apresenta probabilidade significativa ({best_prob:.2%}) de queda superior a {threshold:.0%}.", styles['Normal']))
        elements.append(Paragraph("• Considere implementar proteções como stop-loss ou hedges para mitigar esse risco.", styles['Normal']))
    else:
        elements.append(Paragraph(f"• O ativo apresenta baixa probabilidade ({best_prob:.2%}) de queda superior a {threshold:.0%}.", styles['Normal']))
        elements.append(Paragraph("• Monitorar regularmente para identificar mudanças no padrão de risco.", styles['Normal']))
    
    # Rodapé
    elements.append(Spacer(1, 0.5*inch))
    footnote_style = ParagraphStyle(
        name='Footnote',
        fontSize=8,
        alignment=1
    )
    elements.append(Paragraph("Análise gerada automaticamente pelo Sistema Bilionário", footnote_style))
    elements.append(Paragraph(f"Gerado em: {datetime.now().strftime('%d/%m/%Y às %H:%M')}", footnote_style))
    
    # Construir PDF
    pdf.build(elements)
    return buffer.getvalue()

def generate_complete_statistical_analysis_pdf(df, asset_symbol=None):
    """
    Gera um PDF completo com dados reais da análise estatística
    
    Args:
        df: DataFrame com os dados dos ativos
        asset_symbol: Símbolo do ativo principal para análise
        
    Returns:
        bytes: Conteúdo do PDF
    """
    import matplotlib.pyplot as plt
    import io
    from reportlab.lib.utils import ImageReader
    import matplotlib
    matplotlib.use('Agg')  # Use Agg backend para não mostrar plots
    
    # Se não for especificado um símbolo, usar o primeiro disponível ou PETR4.SA
    if asset_symbol is None:
        try:
            from constants import PETR4_SYMBOL
            asset_symbol = PETR4_SYMBOL
        except:
            if df is not None and not df.empty:
                asset_symbol = df.columns[0]
            else:
                asset_symbol = "ATIVO"
    
    buffer = BytesIO()
    pdf = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []
    
    # Título principal e data
    title_style = ParagraphStyle(
        name="Title",
        fontSize=22,
        alignment=1,
        spaceAfter=12
    )
    
    elements.append(Paragraph(f"Análise Estatística Completa", title_style))
    elements.append(Spacer(1, 0.2*inch))
    elements.append(Paragraph(f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}", styles['Normal']))
    elements.append(Spacer(1, 0.3*inch))
    
    # Calcular estatísticas básicas se tiver dados
    if df is not None and not df.empty and asset_symbol in df.columns:
        # Obter os dados do ativo selecionado
        asset_data = df[asset_symbol].dropna()
        asset_returns = asset_data.pct_change().dropna()
        
        # Estatísticas descritivas
        elements.append(Paragraph(f"Análise do ativo: {asset_symbol}", styles['Heading2']))
        elements.append(Spacer(1, 0.1*inch))
        
        stats_data = [
            ["Estatística", "Valor"],
            ["Preço Atual", f"$ {asset_data.iloc[-1]:.2f}"],
            ["Retorno Médio", f"{asset_returns.mean():.4%}"],
            ["Volatilidade (diária)", f"{asset_returns.std():.4%}"],
            ["Volatilidade (anual)", f"{asset_returns.std() * np.sqrt(252):.4%}"],
            ["Mínimo", f"{asset_returns.min():.2%}"],
            ["Máximo", f"{asset_returns.max():.2%}"],
            ["Assimetria", f"{skew(asset_returns):.4f}"],
            ["Curtose", f"{kurtosis(asset_returns):.4f}"]
        ]
        
        stats_table = Table(stats_data, colWidths=[2.5*inch, 2.5*inch])
        stats_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        
        elements.append(stats_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Adicionar gráficos
        try:
            # Gráfico de preços
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(asset_data.index, asset_data.values)
            ax.set_title(f'Evolução do Preço: {asset_symbol}')
            ax.set_xlabel('Data')
            ax.set_ylabel('Preço')
            plt.tight_layout()
              # Salvar o gráfico em um buffer
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=100)
            img_buffer.seek(0)
            
            # Adicionar o gráfico ao PDF
            elements.append(Paragraph("Evolução de Preços", styles['Heading3']))
            elements.append(Image(img_buffer, width=6*inch, height=3*inch))
            elements.append(Spacer(1, 0.2*inch))
            
            plt.close(fig)
            
            # Gráfico de distribuição de retornos
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(asset_returns, bins=30, alpha=0.7, density=True)
            ax.set_title(f'Distribuição de Retornos: {asset_symbol}')
            ax.set_xlabel('Retorno Diário')
            ax.set_ylabel('Densidade')
            plt.tight_layout()
            # Salvar o gráfico em um buffer
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=100)
            img_buffer.seek(0)
            
            # Adicionar o gráfico ao PDF
            elements.append(Paragraph("Distribuição de Retornos", styles['Heading3']))
            elements.append(Image(img_buffer, width=6*inch, height=3*inch))
            
            plt.close(fig)
        except Exception as e:
            elements.append(Paragraph(f"Erro ao gerar gráficos: {str(e)}", styles['Normal']))
    else:
        elements.append(Paragraph("Dados insuficientes para análise estatística.", styles['Normal']))
    
    # Adicionar seções temáticas - Eventos Extremos
    elements.append(Spacer(1, 0.3*inch))
    elements.append(Paragraph("Análise de Eventos Extremos", styles['Heading2']))
    elements.append(Spacer(1, 0.1*inch))
    
    if df is not None and not df.empty and asset_symbol in df.columns:
        # Calcular probabilidades empíricas para diferentes thresholds
        thresholds = [0.02, 0.05, 0.10]
        prob_data = [["Threshold", "Probabilidade Empírica", "# de Ocorrências"]]
        
        for threshold in thresholds:
            extreme_count = len(asset_returns[asset_returns < -threshold])
            prob = extreme_count / len(asset_returns) if len(asset_returns) > 0 else 0
            prob_data.append([f"{threshold:.0%}", f"{prob:.2%}", f"{extreme_count}"])
            
        # Criar tabela de probabilidades
        prob_table = Table(prob_data, colWidths=[1.5*inch, 2*inch, 1.5*inch])
        prob_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        
        elements.append(Paragraph("Probabilidades de Quedas Extremas:", styles['Normal']))
        elements.append(Spacer(1, 0.1*inch))
        elements.append(prob_table)
        elements.append(Spacer(1, 0.1*inch))
        elements.append(Paragraph("Avaliação do risco de eventos extremos baseada em dados históricos.", styles['Normal']))
    else:
        elements.append(Paragraph("Dados insuficientes para análise de eventos extremos.", styles['Normal']))
    
    # Adicionar seções temáticas - Modelos de Risco
    elements.append(Spacer(1, 0.3*inch))
    elements.append(Paragraph("Modelos de Risco", styles['Heading2']))
    elements.append(Spacer(1, 0.1*inch))
    
    if df is not None and not df.empty and asset_symbol in df.columns:
        # Calcular métricas de risco simplificadas
        try:
            # Value at Risk
            var_95 = np.percentile(asset_returns, 5)
            var_99 = np.percentile(asset_returns, 1)
            cvar_95 = asset_returns[asset_returns <= var_95].mean() if len(asset_returns[asset_returns <= var_95]) > 0 else var_95
            
            # Volatilidade Anualizada
            vol_annual = asset_returns.std() * np.sqrt(252)
            
            # Drawdown simples
            cumulative_returns = (1 + asset_returns).cumprod()
            max_returns = cumulative_returns.cummax()
            drawdowns = (cumulative_returns / max_returns) - 1
            max_drawdown = drawdowns.min()
            
            # Criar tabela de métricas de risco
            risk_data = [
                ["Métrica", "Valor", "Interpretação"],
                ["VaR (95%)", f"{var_95:.2%}", f"Perda máxima diária com 95% de confiança"],
                ["VaR (99%)", f"{var_99:.2%}", f"Perda máxima diária com 99% de confiança"],
                ["CVaR (95%)", f"{cvar_95:.2%}", f"Perda média nos 5% piores dias"],
                ["Vol. Anual", f"{vol_annual:.2%}", f"Volatilidade anualizada do ativo"],
                ["Max Drawdown", f"{max_drawdown:.2%}", f"Maior queda histórica do pico ao vale"]
            ]
            
            risk_table = Table(risk_data, colWidths=[1.5*inch, 1.5*inch, 2*inch])
            risk_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            
            elements.append(risk_table)
            
            # Classificação de risco
            elements.append(Spacer(1, 0.2*inch))
            
            # Definir nível de risco baseado na volatilidade anualizada
            risk_level = "BAIXO"
            risk_color = "Verde"
            if vol_annual > 0.50:  # >50%
                risk_level = "MUITO ALTO"
                risk_color = "Vermelho"
            elif vol_annual > 0.30:  # >30%
                risk_level = "ALTO"
                risk_color = "Laranja"
            elif vol_annual > 0.20:  # >20%
                risk_level = "MODERADO"
                risk_color = "Amarelo"
                
            elements.append(Paragraph(f"Classificação de Risco: {risk_level}", styles['Heading3']))
            elements.append(Paragraph(f"Baseado na volatilidade anualizada de {vol_annual:.2%}, este ativo apresenta risco {risk_level.lower()}.", styles['Normal']))
            
        except Exception as e:
            elements.append(Paragraph(f"Erro ao calcular métricas de risco: {str(e)}", styles['Normal']))
    else:
        elements.append(Paragraph("Dados insuficientes para análise de risco.", styles['Normal']))
    
    # Pair Trading
    elements.append(Spacer(1, 0.3*inch))
    elements.append(Paragraph("Pair Trading Avançado", styles['Heading2']))
    elements.append(Spacer(1, 0.1*inch))
    
    elements.append(Paragraph("A análise de pair trading identifica pares de ativos com potencial para estratégias de negociação estatística baseadas em cointegração.", styles['Normal']))
    elements.append(Spacer(1, 0.1*inch))
    elements.append(Paragraph("Para análises detalhadas de pares específicos, utilize a interface interativa da aplicação.", styles['Normal']))
    
    # Rodapé
    elements.append(Spacer(1, 1*inch))
    footnote_style = ParagraphStyle(
        name='Footnote',
        fontSize=8,
        alignment=1
    )
    elements.append(Paragraph("Análise gerada automaticamente pelo Sistema Bilionário", footnote_style))
    elements.append(Paragraph(f"Gerado em: {datetime.now().strftime('%d/%m/%Y às %H:%M')}", footnote_style))
    
    # Construir PDF
    pdf.build(elements)
    return buffer.getvalue()

def add_download_buttons_to_extreme_analysis(selected_asset, threshold, prob_empirical, prob_normal, prob_t, df_param):
    """Adiciona botão de download para análise de eventos extremos"""
    # Criar chaves únicas para este ativo e threshold
    pdf_gen_key = f"pdf_generated_extreme_{selected_asset}_{threshold}"
    pdf_error_key = f"pdf_error_extreme_{selected_asset}_{threshold}"
    
    # Inicializar o estado se necessário
    if pdf_gen_key not in st.session_state:
        st.session_state[pdf_gen_key] = None
    
    st.markdown("### 📥 Exportar Análise")
    
    # Criar um container para mensagens de status
    pdf_container = st.container()
    
    # Usar um formulário para evitar recarregamento completo da página
    with st.form(key=f"pdf_form_extreme_{selected_asset}_{threshold}"):
        st.caption(f"Clique abaixo para preparar o PDF com a análise de eventos extremos para {selected_asset}")
        submit_button = st.form_submit_button(label="📄 Preparar PDF", use_container_width=True)
        
        if submit_button:
            with st.spinner("Gerando PDF..."):
                try:
                    # Gerar PDF diretamente dentro do contexto do formulário
                    pdf_data = generate_extreme_analysis_pdf(
                        selected_asset, threshold, prob_empirical, prob_normal, prob_t, df_param
                    )
                    
                    # Armazenar dados no session_state
                    st.session_state[pdf_gen_key] = {
                        "pdf_data": pdf_data,
                        "asset": selected_asset,
                        "threshold": threshold,
                        "ready": True
                    }
                    
                    # Não mostramos mensagem de sucesso aqui porque o formulário
                    # ainda vai recarregar a página, mas agora o PDF estará no session_state
                except Exception as e:
                    st.session_state[pdf_error_key] = str(e)
    
    # Verificar se ocorreu algum erro durante a geração
    if pdf_error_key in st.session_state and st.session_state[pdf_error_key]:
        with pdf_container:
            st.error(f"Erro ao gerar PDF: {st.session_state[pdf_error_key]}")
            # Limpar erro após mostrar
            st.session_state[pdf_error_key] = None
    
    # Verificar se o PDF foi gerado com sucesso
    if st.session_state.get(pdf_gen_key) and st.session_state[pdf_gen_key].get("ready", False):
        pdf_info = st.session_state[pdf_gen_key]
        pdf_data = pdf_info["pdf_data"]
        asset = pdf_info["asset"]
        
        with pdf_container:
            st.success("PDF gerado com sucesso! Clique abaixo para baixar.")
        
        # Mostrar botão de download
        download_button = create_download_button(
            pdf_data,
            f"analise_extremos_{asset.replace('.', '_')}.pdf", 
            "⬇️ Baixar PDF"
        )
        st.markdown(download_button, unsafe_allow_html=True)
    else:
        # Mostrar dica quando não houver PDF gerado ainda
        with pdf_container:
            if not submit_button:  # Não mostrar quando acabamos de clicar no botão
                st.info("Clique no botão acima para preparar o PDF antes de baixar")
            
def generate_distribution_comparison_pdf(asset1, asset2, comparison_tests, descriptive_stats, data_points):
    """Gera PDF para comparação de distribuições estatísticas
    
    Args:
        asset1 (str): Nome do primeiro ativo
        asset2 (str): Nome do segundo ativo
        comparison_tests (dict): Resultados dos testes estatísticos
        descriptive_stats (dict): Estatísticas descritivas dos dois ativos
        data_points (dict): Número de pontos de dados para cada ativo
    
    Returns:
        bytes: Conteúdo do PDF
    """
    import matplotlib.pyplot as plt
    import io
    
    buffer = BytesIO()
    pdf = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []
    
    # Título e cabeçalho
    title_style = ParagraphStyle(
        name="Title",
        fontSize=18,
        alignment=1,
        spaceAfter=12
    )
    
    elements.append(Paragraph(f"Comparação Estatística de Distribuições", title_style))
    elements.append(Spacer(1, 0.2*inch))
    elements.append(Paragraph(f"{asset1} vs {asset2}", styles['Heading2']))
    elements.append(Spacer(1, 0.1*inch))
    elements.append(Paragraph(f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M')}", styles['Normal']))
    elements.append(Spacer(1, 0.3*inch))
    
    # Sumário dos dados
    elements.append(Paragraph("Resumo dos Dados", styles['Heading2']))
    elements.append(Spacer(1, 0.1*inch))
    
    data_summary = [
        ["Ativo", "Observações"],
        [asset1, str(data_points.get(asset1, "N/A"))],
        [asset2, str(data_points.get(asset2, "N/A"))]
    ]
    
    summary_table = Table(data_summary, colWidths=[3*inch, 2*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))
    
    elements.append(summary_table)
    elements.append(Spacer(1, 0.2*inch))
    
    # Resultados dos Testes Estatísticos
    elements.append(Paragraph("Resultados dos Testes Estatísticos", styles['Heading2']))
    elements.append(Spacer(1, 0.1*inch))
    
    # Verificar se temos testes estatísticos
    if comparison_tests:
        test_data = [["Teste", "Estatística", "P-valor", "Significativo"]]
        
        # Teste KS
        if 'ks_test' in comparison_tests:
            ks_test = comparison_tests['ks_test']
            test_data.append([
                "Kolmogorov-Smirnov", 
                f"{ks_test.get('statistic', 'N/A'):.4f}", 
                f"{ks_test.get('p_value', 'N/A'):.4f}", 
                "✓" if ks_test.get('significant', False) else "✗"
            ])
        
        # Mann-Whitney U
        if 'mann_whitney' in comparison_tests:
            mw_test = comparison_tests['mann_whitney']
            test_data.append([
                "Mann-Whitney U", 
                f"{mw_test.get('statistic', 'N/A'):.4f}", 
                f"{mw_test.get('p_value', 'N/A'):.4f}", 
                "✓" if mw_test.get('significant', False) else "✗"
            ])
        
        # Levene
        if 'levene' in comparison_tests:
            levene_test = comparison_tests['levene']
            test_data.append([
                "Levene", 
                f"{levene_test.get('statistic', 'N/A'):.4f}", 
                f"{levene_test.get('p_value', 'N/A'):.4f}",
                "✓" if levene_test.get('significant', False) else "✗"
            ])
        
        test_table = Table(test_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1*inch])
        test_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        
        elements.append(test_table)
    else:
        elements.append(Paragraph("Não há resultados de testes estatísticos disponíveis.", styles['Normal']))
    
    elements.append(Spacer(1, 0.2*inch))
    
    # Interpretação dos Testes
    # Determinar se são significativamente diferentes
    are_different = False
    if comparison_tests:
        ks_significant = comparison_tests.get('ks_test', {}).get('significant', False)
        mw_significant = comparison_tests.get('mann_whitney', {}).get('significant', False)
        are_different = ks_significant or mw_significant
    
    elements.append(Paragraph("Interpretação", styles['Heading2']))
    elements.append(Spacer(1, 0.1*inch))
    
    if are_different:
        elements.append(Paragraph(f"✅ Os ativos {asset1} e {asset2} têm distribuições estatisticamente diferentes.", styles['Normal']))
        elements.append(Paragraph("Isto indica que eles têm comportamentos de mercado distintos, o que pode ser relevante para:", styles['Normal']))
        elements.append(Spacer(1, 0.1*inch))
        elements.append(Paragraph("• Diversificação de portfolio: ativos com comportamentos diferentes ajudam na diversificação", styles['Normal']))
        elements.append(Paragraph("• Pair trading: confirme também a cointegração para estratégias de pair trading", styles['Normal']))
        elements.append(Paragraph("• Alocação de risco: o ativo com maior volatilidade deve receber menor alocação", styles['Normal']))
    else:
        elements.append(Paragraph(f"ℹ️ Os ativos {asset1} e {asset2} não apresentaram diferença estatisticamente significativa.", styles['Normal']))
        elements.append(Paragraph("Isso sugere comportamentos similares em termos de distribuição de retornos:", styles['Normal']))
        elements.append(Spacer(1, 0.1*inch))
        elements.append(Paragraph("• Diversificação limitada: estes ativos podem oferecer menos benefícios de diversificação", styles['Normal']))
        elements.append(Paragraph("• Correlação: verifique a correlação entre eles para entender se movem juntos", styles['Normal']))
        elements.append(Paragraph("• Análise setorial: podem pertencer ao mesmo setor ou ser afetados pelos mesmos fatores", styles['Normal']))
    
    elements.append(Spacer(1, 0.3*inch))
    
    # Tabelas de Estatísticas Descritivas
    elements.append(Paragraph("Estatísticas Descritivas", styles['Heading2']))
    elements.append(Spacer(1, 0.1*inch))
    
    # Verificar se temos estatísticas descritivas
    if descriptive_stats:
        # Tabela para o primeiro ativo
        if asset1 in descriptive_stats:
            elements.append(Paragraph(f"Ativo: {asset1}", styles['Heading3']))
            elements.append(Spacer(1, 0.1*inch))
            
            asset1_stats = descriptive_stats[asset1]
            stats_data1 = [
                ["Métrica", "Valor"],
                ["Média", f"{asset1_stats.get('mean', 'N/A'):.5f}"],
                ["Mediana", f"{asset1_stats.get('median', 'N/A'):.5f}"],
                ["Desvio Padrão", f"{asset1_stats.get('std', 'N/A'):.5f}"],
                ["Assimetria", f"{asset1_stats.get('skew', 'N/A'):.5f}"],
                ["Curtose", f"{asset1_stats.get('kurtosis', 'N/A'):.5f}"]
            ]
            
            stats_table1 = Table(stats_data1, colWidths=[2*inch, 2*inch])
            stats_table1.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            
            elements.append(stats_table1)
            elements.append(Spacer(1, 0.2*inch))
        
        # Tabela para o segundo ativo
        if asset2 in descriptive_stats:
            elements.append(Paragraph(f"Ativo: {asset2}", styles['Heading3']))
            elements.append(Spacer(1, 0.1*inch))
            
            asset2_stats = descriptive_stats[asset2]
            stats_data2 = [
                ["Métrica", "Valor"],
                ["Média", f"{asset2_stats.get('mean', 'N/A'):.5f}"],
                ["Mediana", f"{asset2_stats.get('median', 'N/A'):.5f}"],
                ["Desvio Padrão", f"{asset2_stats.get('std', 'N/A'):.5f}"],
                ["Assimetria", f"{asset2_stats.get('skew', 'N/A'):.5f}"],
                ["Curtose", f"{asset2_stats.get('kurtosis', 'N/A'):.5f}"]
            ]
            
            stats_table2 = Table(stats_data2, colWidths=[2*inch, 2*inch])
            stats_table2.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            
            elements.append(stats_table2)
    else:
        elements.append(Paragraph("Não há estatísticas descritivas disponíveis.", styles['Normal']))
    
    # Rodapé
    elements.append(Spacer(1, 0.5*inch))
    footnote_style = ParagraphStyle(
        name='Footnote',
        fontSize=8,
        alignment=1
    )
    elements.append(Paragraph("Análise gerada automaticamente pelo Sistema Bilionário", footnote_style))
    elements.append(Paragraph(f"Gerado em: {datetime.now().strftime('%d/%m/%Y às %H:%M')}", footnote_style))
    
    # Construir PDF
    pdf.build(elements)
    return buffer.getvalue()

def add_download_button_to_distribution_comparison(asset1, asset2, comparison_tests, descriptive_stats, data_points):
    """Adiciona botão de download para comparação estatística de distribuições
    
    Args:
        asset1 (str): Nome do primeiro ativo
        asset2 (str): Nome do segundo ativo
        comparison_tests (dict): Resultados dos testes estatísticos
        descriptive_stats (dict): Estatísticas descritivas dos dois ativos
        data_points (dict): Número de pontos de dados para cada ativo
    """
    # Criar chaves únicas para este par de ativos
    pdf_gen_key = f"pdf_generated_{asset1}_{asset2}"
    pdf_error_key = f"pdf_error_{asset1}_{asset2}"
    
    # Inicializar o estado se necessário
    if pdf_gen_key not in st.session_state:
        st.session_state[pdf_gen_key] = None
    
    st.markdown("### 📥 Exportar Análise")
    
    # Criar um container para mensagens de status
    pdf_container = st.container()
    
    # Gerar PDF automaticamente quando encontrar pares diferentes
    if st.session_state.get(pdf_gen_key) is None:
        try:
            with st.spinner("Gerando PDF..."):
                # Gerar PDF diretamente
                pdf_data = generate_distribution_comparison_pdf(
                    asset1, asset2, comparison_tests, descriptive_stats, data_points
                )
                
                # Armazenar dados no session_state
                st.session_state[pdf_gen_key] = {
                    "pdf_data": pdf_data,
                    "asset1": asset1,
                    "asset2": asset2,
                    "ready": True
                }
        except Exception as e:
            st.session_state[pdf_error_key] = str(e)
    
    # Verificar se ocorreu algum erro durante a geração
    if pdf_error_key in st.session_state and st.session_state[pdf_error_key]:
        with pdf_container:
            st.error(f"Erro ao gerar PDF: {st.session_state[pdf_error_key]}")
            # Limpar erro após mostrar
            st.session_state[pdf_error_key] = None
    
    # Verificar se o PDF foi gerado com sucesso
    if st.session_state.get(pdf_gen_key) and st.session_state[pdf_gen_key].get("ready", False):
        pdf_info = st.session_state[pdf_gen_key]
        pdf_data = pdf_info["pdf_data"]
        a1 = pdf_info["asset1"]
        a2 = pdf_info["asset2"]
        
        with pdf_container:
            st.success("PDF gerado com sucesso! Clique abaixo para baixar.")
        
        # Mostrar botão de download diretamente
        download_button = create_download_button(
            pdf_data,
            f"comparacao_distribuicoes_{a1.replace('.', '_')}_{a2.replace('.', '_')}.pdf", 
            "⬇️ Baixar PDF"
        )
        st.markdown(download_button, unsafe_allow_html=True)
