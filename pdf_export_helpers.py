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
            img = ImageReader(img_buffer)
            
            # Adicionar o gráfico ao PDF
            elements.append(Paragraph("Evolução de Preços", styles['Heading3']))
            elements.append(Image(img, width=6*inch, height=3*inch))
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
            img = ImageReader(img_buffer)
            
            # Adicionar o gráfico ao PDF
            elements.append(Paragraph("Distribuição de Retornos", styles['Heading3']))
            elements.append(Image(img, width=6*inch, height=3*inch))
            
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
    st.markdown("### 📥 Exportar Análise")
    
    if st.button("📄 Gerar PDF desta Análise", key=f"pdf_{selected_asset}_{threshold}"):
        try:
            with st.spinner("Gerando PDF da análise de eventos extremos..."):
                pdf_data = generate_extreme_analysis_pdf(
                    selected_asset, threshold, prob_empirical, prob_normal, prob_t, df_param
                )
                
                download_button = create_download_button(
                    pdf_data,
                    f"analise_extremos_{selected_asset.replace('.', '_')}.pdf", 
                    "Baixar Análise de Eventos Extremos (PDF)"
                )
                st.markdown(download_button, unsafe_allow_html=True)
                st.success("PDF gerado com sucesso! Clique no botão acima para baixar.")
        except Exception as e:
            st.error(f"Erro ao gerar PDF: {str(e)}")
