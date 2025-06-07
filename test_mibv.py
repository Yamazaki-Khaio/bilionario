import yfinance as yf

# Testar variações do mini Ibovespa
tickers = ['MIBV.SA', 'WINM25.SA', 'WIN1125.SA', 'WIMV25.SA', '^BVSP', 'BOVA11.SA']

for ticker in tickers:
    try:
        data = yf.Ticker(ticker).history(period='5d')
        if not data.empty:
            print(f"{ticker}: ✅ ENCONTRADO - {len(data)} dias")
            print(f"Último preço: {data['Close'].iloc[-1]:.2f}")
        else:
            print(f"{ticker}: ❌ NÃO ENCONTRADO")
    except Exception as e:
        print(f"{ticker}: ❌ ERRO - {str(e)}")
