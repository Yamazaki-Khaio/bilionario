import yfinance as yf

# Testar diferentes variações do ticker WIN (mini futuro Ibovespa)
tickers_to_test = [
    'WIN25.SA',    # WIN + ano
    'WINZ25.SA',   # WIN + Z (dezembro) + ano
    'WING25.SA',   # WIN + G (fevereiro) + ano
    'WIN2025.SA',  # WIN + ano completo
    'WIN24.SA',    # WIN + ano anterior
    'WINZ24.SA',   # WIN + Z + ano anterior
    'WING24.SA',   # WIN + G + ano anterior
    'WINI25.SA',   # WIN + I (setembro) + ano
    'WINJ25.SA',   # WIN + J (abril) + ano
    'WINK25.SA',   # WIN + K (maio) + ano
    'WINM25.SA',   # WIN + M (junho) + ano
    'WINN25.SA',   # WIN + N (julho) + ano
    'WINQ25.SA',   # WIN + Q (agosto) + ano
    'WINU25.SA',   # WIN + U (setembro) + ano
    'WINV25.SA',   # WIN + V (outubro) + ano
    'WINX25.SA',   # WIN + X (novembro) + ano
]

print("Testando tickers do WIN (mini futuro Ibovespa)...")
valid_tickers = []

for ticker in tickers_to_test:
    try:
        data = yf.download(ticker, period="5d", progress=False)
        if len(data) > 0:
            print(f"✅ {ticker}: {len(data)} linhas - VÁLIDO")
            print(f"   Última cotação: {data['Close'].iloc[-1]:.2f}")
            valid_tickers.append(ticker)
        else:
            print(f"❌ {ticker}: Sem dados")
    except Exception as e:
        print(f"❌ {ticker}: Erro - {str(e)}")

print(f"\nTickers válidos encontrados: {valid_tickers}")
