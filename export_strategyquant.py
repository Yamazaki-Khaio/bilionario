import pandas as pd
import pandas_ta as ta

def export_to_strategyquant():
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
    ind_df = pd.DataFrame(index=df.index)
    params_list = []

    for ticker in df.columns:
        series = df[ticker]
        # SMA curto e longo
        for key in ['sma_short', 'sma_long']:
            p = default_params[key]
            ind_df[f"{ticker}_{key}_{p['default']}"] = ta.sma(series, length=p['default'])
            params_list.append({'ticker': ticker, 'indicator': key, **p})
        # RSI
        p = default_params['rsi']
        ind_df[f"{ticker}_rsi_{p['default']}"] = ta.rsi(series, length=p['default'])
        params_list.append({'ticker': ticker, 'indicator': 'rsi', **p})
        # MACD (gera colunas MACD and MACDs)
        pf = default_params['macd_fast']['default']
        ps = default_params['macd_slow']['default']
        psg = default_params['macd_signal']['default']
        macd_df = ta.macd(series, fast=pf, slow=ps, signal=psg)
        for col in macd_df.columns:
            ind_df[f"{ticker}_{col}"] = macd_df[col]
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

if __name__ == '__main__':
    export_to_strategyquant()
