import yfinance as yf
import pandas as pd
import os

# Lista de ativos - Expandida para 20 ações brasileiras
TICKERS = [
    # Ações brasileiras (20 principais)
    'PETR4.SA','VALE3.SA','ITUB4.SA','BBDC4.SA','ABEV3.SA',
    'BBAS3.SA','WEGE3.SA','RENT3.SA','JBSS3.SA','SUZB3.SA',
    'RADL3.SA','LREN3.SA','MGLU3.SA','VIVT3.SA','ELET3.SA',
    'SANB11.SA','CSAN3.SA','USIM5.SA','BRDT3.SA','KLBN11.SA',
    # ETFs e Índices
    'BOVA11.SA','SMAL11.SA','IVVB11.SA',
    # Mini Índice Ibovespa (WI$N via ETF)
    'WIN2025.SA',  # Mini índice Ibovespa
    # Criptomoedas
    'BTC-USD','ETH-USD','ADA-USD','SOL-USD','XRP-USD',
    # Forex
    'USDBRL=X','EURBRL=X','JPY=X','USDJPY=X'
]

# Categorização por setores
ASSET_CATEGORIES = {
    'acoes': ['PETR4.SA','VALE3.SA','ITUB4.SA','BBDC4.SA','ABEV3.SA',
              'BBAS3.SA','WEGE3.SA','RENT3.SA','JBSS3.SA','SUZB3.SA',
              'RADL3.SA','LREN3.SA','MGLU3.SA','VIVT3.SA','ELET3.SA',
              'SANB11.SA','CSAN3.SA','USIM5.SA','BRDT3.SA','KLBN11.SA'],
    'etfs_indices': ['BOVA11.SA','SMAL11.SA','IVVB11.SA','WIN2025.SA'],
    'criptomoedas': ['BTC-USD','ETH-USD','ADA-USD','SOL-USD','XRP-USD'],
    'forex': ['USDBRL=X','EURBRL=X','JPY=X','USDJPY=X']
}

# Pasta de saída
OUTPUT_DIR = 'data'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Baixar dados ajustados (closing prices)
def fetch_data(start='2020-01-01', end=None):
    # Download com preços ajustados (Close já ajustado em yfinance v0.2+)
    df = yf.download(TICKERS, start=start, end=end, auto_adjust=True)['Close']
    df.to_csv(os.path.join(OUTPUT_DIR, 'raw_data.csv'))
    print(f"Dados salvos em {OUTPUT_DIR}/raw_data.csv")

if __name__ == '__main__':
    fetch_data()
