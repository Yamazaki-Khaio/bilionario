#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Funções auxiliares para análise de risco
"""
import streamlit as st
import pandas as pd
import numpy as np

def _display_risk_interpretation(risk_metrics):
    """Exibe interpretação dos resultados da análise de risco"""
    st.subheader("💡 Interpretação dos Resultados de Risco")
    
    # Obter valores das métricas principais
    var_95 = risk_metrics.get('var_cvar', {}).get('var_95', 0)
    var_99 = risk_metrics.get('var_cvar', {}).get('var_99', 0)
    cvar_95 = risk_metrics.get('var_cvar', {}).get('cvar_95', 0)
    
    # Métricas adicionais
    max_dd = risk_metrics.get('risk_metrics', {}).get('max_drawdown', 0)
    vol_anual = risk_metrics.get('volatility_analysis', {}).get('current_vol_annual', 0)
    
    # Criar tabela de interpretação
    interpretation_df = pd.DataFrame({
        "Métrica": ["Value at Risk (95%)", "Conditional VaR (95%)", "Máximo Drawdown", "Volatilidade Anualizada"],
        "Valor": [f"{var_95:.2%}", f"{cvar_95:.2%}", f"{max_dd:.2%}", f"{vol_anual:.2%}"],
        "Interpretação": [
            f"Em 95% dos casos, a perda diária não ultrapassará {abs(var_95):.2%}",
            f"Em caso de evento extremo (5% piores dias), a perda média é de {abs(cvar_95):.2%}",
            f"A maior queda histórica do ativo foi de {abs(max_dd):.2%}",
            f"A volatilidade anualizada do ativo é {vol_anual:.2%}"
        ]
    })
    
    # Mostrar a tabela
    st.table(interpretation_df)
    
    # Avaliação de risco
    risk_score = _calculate_risk_score(var_95, cvar_95, max_dd, vol_anual)
    risk_category = _get_risk_category(risk_score)
    
    # Exibir categoria de risco
    st.markdown(f"### Classificação de Risco: {risk_category['color']} {risk_category['level']}")
    
    # Recomendações
    st.markdown("#### Recomendações:")
    for suggestion in risk_category['suggestions']:
        st.markdown(f"- {suggestion}")


def _calculate_risk_score(var_95, cvar_95, max_dd, vol_anual):
    """Calcula pontuação de risco baseada nas métricas"""
    var_score = min(abs(var_95) * 50, 5)  # Pontuar até 5 baseado no VaR
    cvar_score = min(abs(cvar_95) * 30, 5)  # Pontuar até 5 baseado no CVaR
    dd_score = min(abs(max_dd) * 10, 5)  # Pontuar até 5 baseado no Máximo Drawdown
    vol_score = min(vol_anual * 10, 5)  # Pontuar até 5 baseado na volatilidade
    
    # Média ponderada das pontuações
    risk_score = (var_score * 0.25 + cvar_score * 0.30 + 
                 dd_score * 0.25 + vol_score * 0.20)
    return risk_score


def _get_risk_category(risk_score):
    """Retorna categoria de risco baseada na pontuação"""
    if risk_score <= 1.5:
        return {
            'level': "MUITO BAIXO",
            'color': "🟢",
            'recommendation': "Ativo de baixo risco",
            'suggestions': ["Adequado para perfil conservador", "Pode compor a base do portfólio"]
        }
    elif risk_score <= 2.5:
        return {
            'level': "BAIXO", 
            'color': "🟡",
            'recommendation': "Ativo adequado para perfil moderadamente conservador",
            'suggestions': ["Pode compor 40-60% do portfólio", "Adequado para diversificação"]
        }
    elif risk_score <= 3.5:
        return {
            'level': "MODERADO",
            'color': "🟠", 
            'recommendation': "Ativo de risco equilibrado",
            'suggestions': ["Limite a 30-40% do portfólio", "Implemente stop-loss em -15%"]
        }
    elif risk_score <= 4.5:
        return {
            'level': "ALTO",
            'color': "🔴",
            'recommendation': "Ativo de alto risco, apenas para perfil arrojado", 
            'suggestions': ["Limite a 15-25% do portfólio", "Stop-loss obrigatório"]
        }
    else:
        return {
            'level': "MUITO ALTO",
            'color': "🚫",
            'recommendation': "Ativo de risco extremo",
            'suggestions': ["Máximo 5-10% do portfólio", "Monitoramento intraday necessário"]
        }
