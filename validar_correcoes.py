"""
Script de validação para testar se as correções foram aplicadas corretamente.
"""

import os
import sys
import importlib
import inspect
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Configuração
print("=== Iniciando validação das correções ===")
print(f"Data e hora: {datetime.now()}")
print("Diretório atual:", os.getcwd())

# Funções de validação
def check_selected_assets_usage(module_name, function_name):
    """Verifica se selected_assets está sendo usado corretamente em uma função"""
    try:
        # Importar o módulo dinamicamente
        module = importlib.import_module(module_name)
        # Obter a função
        function = getattr(module, function_name)
        # Obter o código fonte
        source = inspect.getsource(function)
        
        # Verificar se selected_assets é um parâmetro
        sig = inspect.signature(function)
        has_selected_assets = 'selected_assets' in sig.parameters
        
        # Verificar se o parâmetro é usado no código (simplificado)
        uses_selected_assets = "selected_assets" in source and "returns_selected[selected_assets]" in source
        
        print(f"Função {function_name} em {module_name}:")
        print(f"  - Tem parâmetro selected_assets: {has_selected_assets}")
        print(f"  - Usa selected_assets: {uses_selected_assets}")
        
        return has_selected_assets and uses_selected_assets
    except Exception as e:
        print(f"ERRO ao verificar {function_name} em {module_name}: {str(e)}")
        return False

# Verificar correções em advanced_pca_helpers.py
print("\n=== Verificando advanced_pca_helpers.py ===")
functions_to_check = [
    ('advanced_pca_helpers', 'execute_static_pca_analysis'),
    ('advanced_pca_helpers', 'execute_rolling_pca_analysis'),
    ('advanced_pca_helpers', 'build_pca_portfolio')
]

all_passed = True
for module_name, function_name in functions_to_check:
    result = check_selected_assets_usage(module_name, function_name)
    all_passed = all_passed and result

# Verificar correções em advanced_pca_simplificado.py
print("\n=== Verificando advanced_pca_simplificado.py ===")
functions_to_check = [
    ('advanced_pca_simplificado', 'execute_static_pca_analysis'),
    ('advanced_pca_simplificado', 'execute_rolling_pca_analysis'),
    ('advanced_pca_simplificado', 'build_pca_portfolio')
]

for module_name, function_name in functions_to_check:
    result = check_selected_assets_usage(module_name, function_name)
    all_passed = all_passed and result

# Verificar statistical_analysis_helpers.py
print("\n=== Verificando statistical_analysis_helpers.py ===")
try:
    with open('statistical_analysis_helpers.py', 'r') as f:
        content = f.read()
    
    # Verificar se o bug "try" sem "except" foi corrigido
    has_bare_try = "try\n" in content
    print(f"Contém 'try' sem 'except': {has_bare_try}")
    
    all_passed = all_passed and not has_bare_try
except Exception as e:
    print(f"ERRO ao verificar statistical_analysis_helpers.py: {str(e)}")
    all_passed = False

# Resultado final
print("\n=== Resultado da validação ===")
if all_passed:
    print("✅ TODAS AS CORREÇÕES FORAM APLICADAS CORRETAMENTE")
else:
    print("❌ ALGUMAS CORREÇÕES NÃO FORAM APLICADAS CORRETAMENTE")

print("\n=== Fim da validação ===")
