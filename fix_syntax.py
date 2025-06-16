"""
Script temporário para corrigir erro de sintaxe no arquivo pair_trading_advanced.py
"""

with open('pair_trading_advanced.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Corrigir o erro de sintaxe
corrected_content = content.replace('if annual_vol > 0 :', 'if annual_vol > 0 else')

with open('pair_trading_advanced.py', 'w', encoding='utf-8') as f:
    f.write(corrected_content)

print("Arquivo pair_trading_advanced.py corrigido com sucesso!")
