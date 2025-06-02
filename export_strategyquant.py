import numpy as np
np.NaN = np.nan
import pandas as pd

# Indicadores técnicos implementados em pandas
def sma(series, length):
    return series.rolling(length).mean()

def rsi(series, length):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=length-1, min_periods=length).mean()
    avg_loss = loss.ewm(com=length-1, min_periods=length).mean()
    rs = avg_gain / avg_loss
    return 100 - 100/(1 + rs)

def macd(series, fast, slow, signal):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def export_to_excel():
    # Carrega dados brutos
    df = pd.read_csv('data/raw_data.csv', index_col=0, parse_dates=True)
    df.fillna(method='ffill', inplace=True)

    # Parâmetros padrão para indicadores técnicos
    default_params = {
        'sma_short':   {'default': 10,  'min': 5,   'max': 30},
        'sma_long':    {'default': 50,  'min': 20,  'max': 200},
        'rsi':         {'default': 14,  'min': 5,   'max': 30},
        'macd_fast':   {'default': 12,  'min': 6,   'max': 24},
        'macd_slow':   {'default': 26,  'min': 12,  'max': 52},
        'macd_signal': {'default': 9,   'min': 5,   'max': 20}
    }
    ind_df = pd.DataFrame(index=df.index)  # noqa: F841
    params_list = []

    for ticker in df.columns:
        series = df[ticker]
        # SMA curto e longo
        for key in ['sma_short', 'sma_long']:
            p = default_params[key]
            ind_df[f"{ticker}_{key}_{p['default']}"] = sma(series, p['default'])
            params_list.append({'ticker': ticker, 'indicator': key, **p})
        # RSI
        p = default_params['rsi']
        ind_df[f"{ticker}_rsi_{p['default']}"] = rsi(series, p['default'])
        params_list.append({'ticker': ticker, 'indicator': 'rsi', **p})
        # MACD inline sem variáveis intermediárias
        pf = default_params['macd_fast']['default']
        ps = default_params['macd_slow']['default']
        psg = default_params['macd_signal']['default']
        ind_df[f"{ticker}_macd_{pf}_{ps}_{psg}"] = macd(series, pf, ps, psg)[0]
        ind_df[f"{ticker}_macd_signal_{pf}_{ps}_{psg}"] = macd(series, pf, ps, psg)[1]
        # parâmetros individuais de MACD
        for comp in ['macd_fast', 'macd_slow', 'macd_signal']:
            params_list.append({'ticker': ticker, 'indicator': comp, **default_params[comp]})

    # Concatena preços e indicadores e exporta para Excel
    out_df = pd.concat([df, ind_df], axis=1)
    out_df.to_excel('strategyquant_data.xlsx', engine='openpyxl')
    print('Dados + indicadores exportados para strategyquant_data.xlsx')

    # Exporta parâmetros para CSV (StrategyQuant configs)
    params_df = pd.DataFrame(params_list)
    params_df.to_csv('strategyquant_params.csv', index=False)
    print('Parâmetros de indicadores exportados para strategyquant_params.csv')

    # Configurações de trade e gestão de capital
    trade_configs = []
    tickers = df.columns.tolist()
    n = len(tickers)
    for ticker in tickers:
        trade_configs.append({
            'ticker': ticker,
            'trade_direction': 'long_only',           # tipo de trade: 'long_only' ou 'long_short'
            'entry_block': 'simple_moving_average',   # bloco de entrada: sinal/indicador
            'entry_order': 'limit',                   # tipo de ordem de entrada: 'limit' ou 'stop'
            'entry_offset': 0.001,                    # offset (ex: 0.1%) para ordem limit
            'exit_type': 'percentage',                # tipo de saída: 'percentage' ou 'trailing_stop'
            'exit_target': 0.05,                      # alvo de lucro (5%)
            'exit_stop': 0.02,                        # stop loss (2%)
            'order_type': 'market',                   # tipo de execução: 'market' ou 'limit'
            'capital_allocation': round(1.0/n, 4)     # alocação de capital (porcentagem)
        })
    trade_df = pd.DataFrame(trade_configs)
    trade_df.to_csv('strategyquant_trade_config.csv', index=False)
    print('Configurações de trade exportadas para strategyquant_trade_config.csv')

if __name__ == '__main__':
    export_to_excel()
