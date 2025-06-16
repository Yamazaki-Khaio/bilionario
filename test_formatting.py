#!/usr/bin/env python3
"""
Teste das funções de formatação financeira
"""

from financial_formatting import (
    format_percentage, format_currency, format_ratio, 
    auto_format_metric, format_metrics_dict, FINANCIAL_METRICS_FORMAT
)
import numpy as np

def test_formatting_functions():
    """Testa todas as funções de formatação"""
    
    print("🔬 TESTE DAS FUNÇÕES DE FORMATAÇÃO FINANCEIRA")
    print("=" * 60)
    
    # 1. Teste format_percentage
    print("\n📊 1. Teste format_percentage:")
    test_values = [0.15, 0.0523, -0.08, 1.25, 0, None, np.nan]
    for val in test_values:
        result = format_percentage(val)
        print(f"  {val} -> {result}")
    
    # 2. Teste format_currency
    print("\n💰 2. Teste format_currency:")
    test_values = [1000.50, 25000, -500.75, 0, None, np.nan]
    for val in test_values:
        result = format_currency(val)
        print(f"  {val} -> {result}")
    
    # 3. Teste format_ratio
    print("\n📈 3. Teste format_ratio:")
    test_values = [1.85, 0.65, -0.25, 0, None, np.nan]
    for val in test_values:
        result = format_ratio(val)
        print(f"  {val} -> {result}")
    
    # 4. Teste auto_format_metric
    print("\n🤖 4. Teste auto_format_metric:")
    metrics = {
        'annual_return': 0.125,
        'volatility': 0.18,
        'max_drawdown': -0.15,
        'sharpe_ratio': 1.45,
        'net_profit': 50000,
        'total_trades': 150
    }
    
    for metric, value in metrics.items():
        result = auto_format_metric(metric, value)
        print(f"  {metric}: {value} -> {result}")
    
    # 5. Teste format_metrics_dict
    print("\n📋 5. Teste format_metrics_dict:")
    sample_metrics = {
        'Retorno Anual': 0.15,
        'Volatilidade': 0.20,
        'Max Drawdown': -0.12,
        'Sharpe Ratio': 1.25,
        'Capital Final': 125000,
        'Número de Trades': 85
    }
    
    formatted = format_metrics_dict(sample_metrics)
    print("  Métricas originais vs formatadas:")
    for key in sample_metrics:
        print(f"    {key}: {sample_metrics[key]} -> {formatted[key]}")
    
    # 6. Teste FINANCIAL_METRICS_FORMAT
    print("\n🎯 6. Métricas suportadas:")
    print(f"  Total de métricas conhecidas: {len(FINANCIAL_METRICS_FORMAT)}")
    
    percentage_metrics = [k for k, v in FINANCIAL_METRICS_FORMAT.items() if v == format_percentage]
    ratio_metrics = [k for k, v in FINANCIAL_METRICS_FORMAT.items() if v == format_ratio]
    currency_metrics = [k for k, v in FINANCIAL_METRICS_FORMAT.items() if v == format_currency]
    
    print(f"  - Métricas de porcentagem: {len(percentage_metrics)}")
    print(f"  - Métricas de ratio: {len(ratio_metrics)}")
    print(f"  - Métricas monetárias: {len(currency_metrics)}")
    
    print("\n✅ TESTE CONCLUÍDO COM SUCESSO!")
    return True

if __name__ == "__main__":
    test_formatting_functions()
