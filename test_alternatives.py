import yfinance as yf

# Testar alternativas ao WIN - índice Ibovespa e ETFs relacionados
alternatives = [
    '^BVSP',      # Índice Ibovespa
    'IBOV.SA',    # Tentativa com .SA
    'IBOV',       # Índice sem .SA
    'IFIX.SA',    # Índice de fundos imobiliários
    'ICON.SA',    # Índice de consumo
    'INDX.SA',    # Índice industrial
    'SMLL.SA',    # Small caps
    'PIBB11.SA',  # ETF que replica o IBrX-100
    'ECOO11.SA',  # ETF de economia circular
    'FIND11.SA',  # ETF financeiro
]

print("Testando alternativas ao WIN (índices e ETFs relacionados)...")
valid_tickers = []

for ticker in alternatives:
    try:
        data = yf.download(ticker, period="5d", progress=False)
        if len(data) > 0:
            print(f"✅ {ticker}: {len(data)} linhas - VÁLIDO")
            try:
                last_price = data['Close'].iloc[-1]
                print(f"   Última cotação: {last_price:.2f}")
            except:
                print(f"   Última cotação: {data['Close'].iloc[-1]}")
            valid_tickers.append(ticker)
        else:
            print(f"❌ {ticker}: Sem dados")
    except Exception as e:
        print(f"❌ {ticker}: Erro - {str(e)}")

print(f"\nTickers válidos encontrados: {valid_tickers}")

# Se encontrar o ^BVSP, mostrar mais detalhes
if '^BVSP' in valid_tickers:
    print("\n📊 Detalhes do IBOVESPA (^BVSP):")
    data = yf.download('^BVSP', period="1mo", progress=False)
    print(f"Dados disponíveis: {len(data)} dias")
    print(f"Período: {data.index[0].strftime('%d/%m/%Y')} a {data.index[-1].strftime('%d/%m/%Y')}")
    print(f"Valor atual: {data['Close'].iloc[-1]:,.0f} pontos")
