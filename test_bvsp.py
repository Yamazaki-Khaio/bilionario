import yfinance as yf

print("Testando ^BVSP (IBOVESPA)...")

try:
    data = yf.download('^BVSP', period="5d", progress=False)
    if len(data) > 0:
        print("✅ ^BVSP: FUNCIONANDO!")
        print(f"Dados: {len(data)} dias")
        print(f"Último valor: {data['Close'].iloc[-1]:.0f} pontos")
        print(f"Data: {data.index[-1].strftime('%d/%m/%Y')}")
    else:
        print("❌ ^BVSP: Sem dados")
except Exception as e:
    print(f"❌ ^BVSP: Erro - {e}")
