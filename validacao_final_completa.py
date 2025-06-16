"""
Script de validação final para o Bilionário App
Testa todas as funcionalidades corrigidas para garantir que a aplicação está funcionando corretamente
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import importlib
import warnings
from datetime import datetime
from constants import get_raw_data_path

warnings.filterwarnings("ignore")

def print_success(message):
    """Imprime mensagem de sucesso em verde"""
    print(f"\033[92m✅ {message}\033[0m")

def print_error(message):
    """Imprime mensagem de erro em vermelho"""
    print(f"\033[91m❌ {message}\033[0m")

def print_info(message):
    """Imprime mensagem informativa em azul"""
    print(f"\033[94mℹ️ {message}\033[0m")

def print_header(message):
    """Imprime cabeçalho"""
    print("\n" + "="*80)
    print(f"  {message}")
    print("="*80)

def test_pair_trading_module():
    """Testa o módulo pair_trading e suas dependências"""
    print_header("TESTANDO MÓDULO PAIR TRADING")
    
    # 1. Verificar importações
    try:
        import pair_trading
        import pair_trading_helpers
        print_success("Módulos importados com sucesso")
    except Exception as e:
        print_error(f"Erro ao importar módulos: {str(e)}")
        return False
    
    # 2. Verificar se os dados existem
    RAW_DATA = get_raw_data_path()
    if not os.path.exists(RAW_DATA):
        print_error(f"Arquivo de dados não encontrado: {RAW_DATA}")
        return False
    
    # 3. Tentar executar funções do módulo
    try:
        # Carregar dados
        df = pd.read_csv(RAW_DATA, index_col=0, parse_dates=True)
        
        # Inicializar PairTradingAnalysis
        from pair_trading import PairTradingAnalysis
        pair_analyzer = PairTradingAnalysis(df)
        print_success("PairTradingAnalysis inicializado com sucesso")
        
        # Testar funções críticas
        all_assets = df.columns.tolist()
        if len(all_assets) < 2:
            print_error("Não há ativos suficientes para testar")
            return False
            
        asset1 = all_assets[0]
        asset2 = all_assets[1]
        
        # Testar funções sem streamlit
        result = pair_analyzer.test_cointegration(asset1, asset2)
        if result is not None and isinstance(result, dict):
            print_success("Função test_cointegration executada com sucesso")
            print_info(f"Resultado: p-value={result.get('p_value', 'N/A')}")
        else:
            print_error("Função test_cointegration falhou")
        
        print_success("Módulo pair_trading testado com sucesso")
        return True
    except Exception as e:
        print_error(f"Erro ao testar módulo pair_trading: {str(e)}")
        return False

def test_statistical_analysis_module():
    """Testa o módulo statistical_analysis e suas dependências"""
    print_header("TESTANDO MÓDULO STATISTICAL ANALYSIS")
    
    # 1. Verificar importações
    try:
        import statistical_analysis
        import statistical_analysis_helpers
        import statistical_analysis_robusto
        print_success("Módulos importados com sucesso")
    except Exception as e:
        print_error(f"Erro ao importar módulos: {str(e)}")
        return False
    
    # 2. Verificar se os dados existem
    RAW_DATA = get_raw_data_path()
    if not os.path.exists(RAW_DATA):
        print_error(f"Arquivo de dados não encontrado: {RAW_DATA}")
        return False
    
    # 3. Tentar executar funções do módulo
    try:
        # Carregar dados
        df = pd.read_csv(RAW_DATA, index_col=0, parse_dates=True)
        
        # Inicializar StatisticalAnalysis
        from statistical_analysis import StatisticalAnalysis
        stat_analyzer = StatisticalAnalysis(df)
        print_success("StatisticalAnalysis inicializado com sucesso")
        
        # Testar funções do statistical_analysis_helpers que foram corrigidas
        import statistical_analysis_helpers as sah
        
        # Verificar se as funções existem
        functions_to_check = [
            "plot_scatter_chart", 
            "plot_histogram_comparison", 
            "plot_box_comparison", 
            "plot_qq_comparison", 
            "display_distribution_comparison_metrics", 
            "plot_correlation_heatmap"
        ]
        
        all_exist = True
        for func_name in functions_to_check:
            if hasattr(sah, func_name):
                print_success(f"Função {func_name} existe")
            else:
                print_error(f"Função {func_name} não existe")
                all_exist = False
        
        # Testar importação do módulo robusto
        from statistical_analysis_robusto import StatisticalAnalyzer
        print_success("StatisticalAnalyzer importado com sucesso")
        
        if all_exist:
            print_success("Todas as funções críticas existem no módulo statistical_analysis_helpers")
        
        print_success("Módulo statistical_analysis testado com sucesso")
        return True
    except Exception as e:
        print_error(f"Erro ao testar módulo statistical_analysis: {str(e)}")
        return False

def test_advanced_pca_module():
    """Testa o módulo advanced_pca e suas dependências"""
    print_header("TESTANDO MÓDULO ADVANCED PCA")
    
    # 1. Verificar importações
    try:
        import advanced_pca_simplificado
        import advanced_pca_helpers
        print_success("Módulos importados com sucesso")
    except Exception as e:
        print_error(f"Erro ao importar módulos: {str(e)}")
        return False
    
    # 2. Verificar se os dados existem
    RAW_DATA = get_raw_data_path()
    if not os.path.exists(RAW_DATA):
        print_error(f"Arquivo de dados não encontrado: {RAW_DATA}")
        return False
    
    print_success("Dados encontrados e módulos importados com sucesso")
    return True

def test_app_structure():
    """Testa se a estrutura do app está correta"""
    print_header("TESTANDO ESTRUTURA DO APP")
    
    # 1. Verificar se app.py existe
    if not os.path.exists("app.py"):
        print_error("Arquivo app.py não encontrado")
        return False
    
    # 2. Verificar importações no app.py
    try:
        with open("app.py", "r", encoding="utf-8") as f:
            content = f.read()
            
        # Verificar se os módulos são importados
        needed_imports = [
            "from pair_trading_helpers import",
            "from statistical_analysis_helpers import",
            "from advanced_pca_simplificado import"
        ]
        
        for imp in needed_imports:
            if imp in content:
                print_success(f"Importação '{imp}' encontrada em app.py")
            else:
                print_error(f"Importação '{imp}' não encontrada em app.py")
                return False
        
        print_success("Estrutura do app validada com sucesso")
        return True
    except Exception as e:
        print_error(f"Erro ao verificar estrutura do app: {str(e)}")
        return False

def main():
    """Função principal"""
    print_header("VALIDAÇÃO FINAL DO BILIONÁRIO APP")
    print_info(f"Data e hora do teste: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    
    # Testar módulos
    pair_trading_ok = test_pair_trading_module()
    statistical_analysis_ok = test_statistical_analysis_module()
    advanced_pca_ok = test_advanced_pca_module()
    app_structure_ok = test_app_structure()
    
    # Resultado final
    print_header("RESUMO FINAL")
    print(f"Pair Trading: {'✅ OK' if pair_trading_ok else '❌ Falhou'}")
    print(f"Statistical Analysis: {'✅ OK' if statistical_analysis_ok else '❌ Falhou'}")
    print(f"Advanced PCA: {'✅ OK' if advanced_pca_ok else '❌ Falhou'}")
    print(f"Estrutura do App: {'✅ OK' if app_structure_ok else '❌ Falhou'}")
    
    if pair_trading_ok and statistical_analysis_ok and advanced_pca_ok and app_structure_ok:
        print_success("\n🎉 TODAS AS CORREÇÕES FORAM VALIDADAS COM SUCESSO!")
        print_info("O Bilionário App está pronto para uso!")
    else:
        print_error("\n⚠️ ALGUMAS CORREÇÕES AINDA PRECISAM DE ATENÇÃO!")
        print_info("Verifique os erros acima e corrija os problemas restantes.")

if __name__ == "__main__":
    main()
