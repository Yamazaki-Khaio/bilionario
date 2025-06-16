#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teste final simplificado para verificar implementações
"""

import os
import re

def print_success(msg):
    """Imprime mensagem de sucesso"""
    print(f"✅ {msg}")

def print_error(msg):
    """Imprime mensagem de erro"""
    print(f"❌ {msg}")

def print_separator():
    """Imprime separador"""
    print("-" * 80)

def main():
    # Verificar petrobras_extreme_analysis em statistical_analysis.py
    print_separator()
    print("Verificando StatisticalAnalysis.petrobras_extreme_analysis")
    with open('statistical_analysis.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    if 'def petrobras_extreme_analysis' in content:
        print_success("Método petrobras_extreme_analysis encontrado")
    else:
        print_error("Método petrobras_extreme_analysis não encontrado")
    
    # Verificar test_advanced_cointegration em pair_trading_advanced.py
    print_separator()
    print("Verificando PairTradingAdvanced.test_advanced_cointegration")
    with open('pair_trading_advanced.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    if 'def test_advanced_cointegration' in content:
        print_success("Método test_advanced_cointegration encontrado")
    else:
        print_error("Método test_advanced_cointegration não encontrado")
    
    # Verificar compare_cointegration_methods em pair_trading_advanced.py
    print_separator()
    print("Verificando PairTradingAdvanced.compare_cointegration_methods")
    if 'def compare_cointegration_methods' in content:
        print_success("Método compare_cointegration_methods encontrado")
    else:
        print_error("Método compare_cointegration_methods não encontrado")
    
    # Verificar referência ao Prof. Carlos Alberto Rodrigues
    print_separator()
    print("Verificando referências ao Prof. Carlos Alberto Rodrigues")
    
    references_found = 0
    files_to_check = ['README.md', 'statistical_analysis.py', 'pair_trading_advanced.py']
    
    for file_path in files_to_check:
        if not os.path.exists(file_path):
            continue
            
        with open(file_path, 'r', encoding='utf-8') as f:
            file_content = f.read()
            
        if 'Carlos Alberto Rodrigues' in file_content:
            print_success(f"Referência encontrada em {file_path}")
            references_found += 1
    
    if references_found == 0:
        print_error("Nenhuma referência ao Prof. Carlos Alberto Rodrigues encontrada")
    
    print_separator()
    print("Verificação completa!")
    
if __name__ == "__main__":
    main()
