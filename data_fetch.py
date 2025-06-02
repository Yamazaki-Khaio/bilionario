import yfinance as yf
import pandas as pd
import os

# Lista de ativos
TICKERS = [
    'PETR4.SA','VALE3.SA','ITUB4.SA','BBDC4.SA','ABEV3.SA',
    'BBAS3.SA','WEGE3.SA','RENT3.SA','JBSS3.SA','SUZB3.SA',
    'RADL3.SA','LREN3.SA','BOVA11.SA','BTC-USD','ETH-USD',
    'ADA-USD','SOL-USD','XRP-USD','USDBRL=X','EURBRL=X'
]

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
