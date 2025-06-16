"""
Utilitários para formatação de métricas financeiras
"""
import numpy as np

def format_percentage(value, decimals=2):
    """
    Formata um valor como porcentagem
    
    Args:
        value (float): Valor decimal (ex: 0.15 para 15%)
        decimals (int): Número de casas decimais
        
    Returns:
        str: Valor formatado como porcentagem
    """
    if value is None or (isinstance(value, (int, float)) and np.isnan(value)):  # NaN check
        return "N/A"
    
    try:
        return f"{float(value) * 100:.{decimals}f}%"
    except (ValueError, TypeError):
        return "N/A"

def format_currency(value, currency="R$", decimals=2):
    """
    Formata um valor como moeda
    
    Args:
        value (float): Valor monetário
        currency (str): Símbolo da moeda
        decimals (int): Número de casas decimais
        
    Returns:
        str: Valor formatado como moeda
    """
    if value is None or (isinstance(value, (int, float)) and np.isnan(value)):  # NaN check
        return f"{currency} N/A"
    
    try:
        return f"{currency} {float(value):,.{decimals}f}"
    except (ValueError, TypeError):
        return f"{currency} N/A"

def format_ratio(value, decimals=2):
    """
    Formata um ratio (Sharpe, Sortino, etc.)
    
    Args:
        value (float): Valor do ratio
        decimals (int): Número de casas decimais
        
    Returns:
        str: Valor formatado
    """
    if value is None or (isinstance(value, (int, float)) and np.isnan(value)):  # NaN check
        return "N/A"
    
    try:
        return f"{float(value):.{decimals}f}"
    except (ValueError, TypeError):
        return "N/A"

def format_number(value, decimals=0):
    """
    Formata um número com separadores de milhares
    
    Args:
        value (float): Valor numérico
        decimals (int): Número de casas decimais
        
    Returns:
        str: Valor formatado
    """
    if value is None or (isinstance(value, (int, float)) and np.isnan(value)):  # NaN check
        return "N/A"
    
    try:
        return f"{float(value):,.{decimals}f}"
    except (ValueError, TypeError):
        return "N/A"

# Constantes para formatação padrão
PERCENTAGE_FORMAT = "{:.2%}"
CURRENCY_FORMAT = "R$ {:.2f}"
RATIO_FORMAT = "{:.2f}"
LARGE_NUMBER_FORMAT = "{:,.0f}"

# Dicionário de formatação para métricas financeiras
FINANCIAL_METRICS_FORMAT = {
    # Retornos (em porcentagem)
    'total_return': format_percentage,
    'annual_return': format_percentage,
    'monthly_return': format_percentage,
    'daily_return': format_percentage,
    
    # Volatilidades (em porcentagem) 
    'annual_volatility': format_percentage,
    'monthly_volatility': format_percentage,
    'daily_volatility': format_percentage,
    'volatility': format_percentage,
    'vol': format_percentage,
    
    # Drawdowns (em porcentagem)
    'max_drawdown': format_percentage,
    'drawdown': format_percentage,
    'avg_drawdown': format_percentage,
    
    # Probabilidades (em porcentagem)
    'probability': format_percentage,
    'win_rate': format_percentage,
    'loss_rate': format_percentage,
    
    # Ratios (sem formatação especial)
    'sharpe_ratio': format_ratio,
    'sortino_ratio': format_ratio,
    'calmar_ratio': format_ratio,
    'profit_factor': format_ratio,
    'recovery_factor': format_ratio,
    
    # Valores monetários
    'equity': format_currency,
    'capital': format_currency,
    'balance': format_currency,
    'net_profit': format_currency,
    'initial_capital': format_currency,
    
    # Números simples
    'trades': format_number,
    'num_trades': format_number,
    'total_trades': format_number,
    'days': format_number,
    'periods': format_number
}

def auto_format_metric(metric_name, value, **kwargs):
    """
    Formata automaticamente uma métrica baseada em seu nome
    
    Args:
        metric_name (str): Nome da métrica
        value: Valor a ser formatado
        **kwargs: Argumentos adicionais para formatação
        
    Returns:
        str: Valor formatado
    """
    # Normalizar nome da métrica
    normalized_name = metric_name.lower().replace('_', '').replace(' ', '')
    
    # Procurar por padrões conhecidos
    for pattern, formatter in FINANCIAL_METRICS_FORMAT.items():
        if pattern.replace('_', '') in normalized_name:
            return formatter(value, **kwargs)
    
    # Formatação padrão para valores numéricos
    if isinstance(value, (int, float)):
        if abs(value) < 1:
            return format_percentage(value)
        elif abs(value) > 1000:
            return format_number(value)
        else:
            return format_ratio(value)
    
    return str(value)

def format_metrics_dict(metrics_dict):
    """
    Formata um dicionário completo de métricas
    
    Args:
        metrics_dict (dict): Dicionário com métricas financeiras
        
    Returns:
        dict: Dicionário com métricas formatadas
    """
    formatted = {}
    
    for key, value in metrics_dict.items():
        formatted[key] = auto_format_metric(key, value)
    
    return formatted
