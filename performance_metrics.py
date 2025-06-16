import pandas as pd
import numpy as np
from financial_formatting import format_percentage, format_ratio, auto_format_metric

def calculate_metrics(returns, initial_capital=10000):
    """
    Calcula métricas de performance para uma série de retornos.
    
    Args:
        returns (pd.Series): Série de retornos diários
        initial_capital (float): Capital inicial
        
    Returns:
        dict: Dicionário com métricas calculadas (retornos e volatilidades já em porcentagem)
    """
    # Converte para Series se for necessário
    if isinstance(returns, list):
        returns = pd.Series(returns)
    
    # Remove valores NaN
    returns = returns.dropna()
    if len(returns) == 0:
        return {
            'total_return': 0.0,
            'annual_return': 0.0,
            'monthly_return': 0.0,
            'annual_volatility': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0
        }

    # Retorno total
    total_return = (1 + returns).prod() - 1
    
    # Retorno anualizado (assumindo 252 dias úteis por ano)
    trading_days = len(returns)
    years = trading_days / 252
    annual_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
    
    # Retorno mensal médio (assumindo 21 dias úteis por mês)
    months = trading_days / 21
    monthly_return = (1 + total_return) ** (1/months) - 1 if months > 0 else annual_return / 12
    
    # Volatilidade anualizada
    annual_volatility = returns.std() * np.sqrt(252)
      # Drawdown máximo
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Sharpe ratio (assumindo taxa livre de risco = 0)
    sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
    
    return {
        'total_return': total_return,
        'annual_return': annual_return,
        'monthly_return': monthly_return,
        'annual_volatility': annual_volatility,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio
    }

def get_monthly_returns(returns):
    """
    Converte retornos diários para mensais.
    
    Args:
        returns (pd.Series): Série de retornos diários
        
    Returns:
        pd.Series: Retornos mensais
    """
    # Converte para Series se for necessário
    if isinstance(returns, list):
        returns = pd.Series(returns)
    
    # Remove valores NaN
    returns = returns.dropna()
    
    if len(returns) == 0:
        return pd.Series()
    
    # Calcula retornos mensais compostos
    monthly_returns = (1 + returns).resample('ME').prod() - 1
    
    return monthly_returns

def calculate_portfolio_metrics(df, weights=None):
    """
    Calcula métricas para um portfólio de ativos.
    
    Args:
        df (pd.DataFrame): DataFrame com preços dos ativos
        weights (list): Pesos do portfólio (default: igual ponderação)
        
    Returns:
        dict: Métricas do portfólio
    """
    # Calcula retornos
    returns = df.pct_change().dropna()
    
    # Define pesos iguais se não especificado
    if weights is None:
        weights = [1/len(df.columns)] * len(df.columns)
    
    # Retorno do portfólio
    portfolio_returns = (returns * weights).sum(axis=1)
    
    # Calcula métricas
    return calculate_metrics(portfolio_returns)

def analyze_drawdown(returns):
    """
    Análise detalhada de drawdown.
    
    Args:
        returns (pd.Series): Série de retornos
        
    Returns:
        dict: Informações sobre drawdowns
    """
    if len(returns) == 0:
        return {}
    
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    
    # Encontra períodos de drawdown
    is_drawdown = drawdown < 0
    drawdown_periods = []
    
    if is_drawdown.any():
        # Agrupa períodos consecutivos de drawdown
        groups = (is_drawdown != is_drawdown.shift()).cumsum()
        for name, group in drawdown.groupby(groups):
            if group.iloc[0] < 0:  # É um período de drawdown
                drawdown_periods.append({
                    'start': group.index[0],
                    'end': group.index[-1],
                    'duration': len(group),
                    'max_drawdown': group.min()
                })
    
    return {
        'max_drawdown': drawdown.min(),
        'avg_drawdown': drawdown[drawdown < 0].mean() if (drawdown < 0).any() else 0,
        'drawdown_periods': len(drawdown_periods),
        'longest_drawdown': max([p['duration'] for p in drawdown_periods]) if drawdown_periods else 0,
        'recovery_factor': abs(returns.sum() / drawdown.min()) if drawdown.min() < 0 else float('inf')
    }

def risk_return_analysis(returns):
    """
    Análise de risco-retorno.
    
    Args:
        returns (pd.Series): Série de retornos
        
    Returns:
        dict: Métricas de risco-retorno
    """
    if len(returns) == 0:
        return {}
    
    metrics = calculate_metrics(returns)
    
    # Métricas adicionais
    var_95 = returns.quantile(0.05)  # Value at Risk 95%
    cvar_95 = returns[returns <= var_95].mean()  # Conditional VaR
    
    # Skewness e Kurtosis
    skewness = returns.skew()
    kurtosis = returns.kurtosis()
    
    # Ratio de Sortino (downside deviation)
    downside_returns = returns[returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(252)
    sortino_ratio = metrics['annual_return'] / downside_deviation if downside_deviation > 0 else 0
    
    return {
        **metrics,
        'var_95': var_95,
        'cvar_95': cvar_95,
        'skewness': skewness,
        'kurtosis': kurtosis,
        'sortino_ratio': sortino_ratio,
        'calmar_ratio': metrics['annual_return'] / abs(metrics['max_drawdown']) if metrics['max_drawdown'] < 0 else 0
    }

def calculate_correlation_metrics(returns):
    """
    Calcula métricas de correlação entre ativos.
    
    Args:
        returns (pd.DataFrame): DataFrame com retornos dos ativos
        
    Returns:
        dict: Métricas de correlação
    """
    correlation_matrix = returns.corr()
    
    # Correlação média
    upper_triangle = correlation_matrix.where(
        np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
    )
    avg_correlation = upper_triangle.stack().mean()
    
    # Correlação máxima e mínima (excluindo diagonal)
    corr_values = upper_triangle.stack()
    max_correlation = corr_values.max()
    min_correlation = corr_values.min()
    
    return {
        'correlation_matrix': correlation_matrix,
        'avg_correlation': avg_correlation,
        'max_correlation': max_correlation,
        'min_correlation': min_correlation
    }