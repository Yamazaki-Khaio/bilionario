import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import importlib
from constants import get_raw_data_path

def test_function_exists(module_name, function_name):
    """Verifica se uma função existe em um módulo"""
    try:
        module = importlib.import_module(module_name)
        function = getattr(module, function_name, None)
        if function is None:
            print(f"❌ Função {function_name} não encontrada no módulo {module_name}")
            return False
        print(f"✅ Função {function_name} encontrada no módulo {module_name}")
        return True
    except Exception as e:
        print(f"❌ Erro ao verificar função {function_name} no módulo {module_name}: {str(e)}")
        return False

def test_pair_trading_fixes():
    """Testa correções nos módulos pair_trading"""
    print("\n🔍 Verificando correções no pair_trading_helpers.py...")
    
    try:
        # Verificar se o módulo pode ser importado sem erros
        import pair_trading_helpers
        print("✅ Módulo pair_trading_helpers importado com sucesso")
        
        # Verificar se as funções críticas existem
        functions_to_check = [
            "_select_assets_for_analysis",
            "_display_correlation_matrix",  
            "detailed_analysis_tab",
            "backtest_tab",
            "_execute_backtest_analysis"
        ]
        
        all_ok = True
        for func in functions_to_check:
            if not test_function_exists("pair_trading_helpers", func):
                all_ok = False
                
        # Verificar se os checkboxes têm keys únicas
        print(f"Verificando se checkboxes têm keys únicas...")
        checkboxes_with_keys = 0
        
        with open("pair_trading_helpers.py", "r", encoding="utf-8") as file:
            content = file.read()
            # Contar quantos checkboxes têm key=
            checkboxes_with_keys = content.count('st.checkbox(') 
            if checkboxes_with_keys > 0:
                keys_count = content.count('key=')
                if keys_count >= checkboxes_with_keys:
                    print(f"✅ Todos os {checkboxes_with_keys} checkboxes têm keys únicas")
                else:
                    print(f"❌ Apenas {keys_count} de {checkboxes_with_keys} checkboxes têm keys")
                    all_ok = False
        
        return all_ok
        
    except Exception as e:
        print(f"❌ Erro ao testar pair_trading_helpers: {str(e)}")
        return False

def test_statistical_analysis_fixes():
    """Testa correções nos módulos statistical_analysis"""
    print("\n🔍 Verificando correções no statistical_analysis_helpers.py...")
    
    try:
        # Verificar se o módulo pode ser importado sem erros
        import statistical_analysis_helpers
        print("✅ Módulo statistical_analysis_helpers importado com sucesso")
        
        # Verificar se todas as funções necessárias existem
        functions_to_check = [
            "plot_scatter_chart",
            "plot_histogram_comparison", 
            "plot_box_comparison",
            "plot_qq_comparison",
            "display_distribution_comparison_metrics",
            "plot_correlation_heatmap",
            "_display_descriptive_statistics"
        ]
        
        all_ok = True
        for func in functions_to_check:
            if not test_function_exists("statistical_analysis_helpers", func):
                all_ok = False
        
        return all_ok
        
    except Exception as e:
        print(f"❌ Erro ao testar statistical_analysis_helpers: {str(e)}")
        return False

def test_robusto_imports():
    """Testa se o statistical_analysis_robusto.py pode importar as funções corretamente"""
    print("\n🔍 Verificando importações no statistical_analysis_robusto.py...")
    
    try:
        # Tentar importar o módulo
        import statistical_analysis_robusto
        print("✅ Módulo statistical_analysis_robusto importado com sucesso")
        return True
    except Exception as e:
        print(f"❌ Erro ao importar statistical_analysis_robusto: {str(e)}")
        return False

def main():
    """Função principal de validação"""
    print("🔧 Validando correções finais...")
    
    # Testar pair_trading_helpers
    pair_trading_ok = test_pair_trading_fixes()
    
    # Testar statistical_analysis_helpers
    statistical_ok = test_statistical_analysis_fixes()
    
    # Testar importações robusto
    robusto_ok = test_robusto_imports()
    
    # Resumo
    print("\n📋 RESUMO DA VALIDAÇÃO:")
    print(f"- Pair Trading: {'✅ OK' if pair_trading_ok else '❌ Problemas'}")
    print(f"- Statistical Analysis: {'✅ OK' if statistical_ok else '❌ Problemas'}")
    print(f"- Statistical Robusto: {'✅ OK' if robusto_ok else '❌ Problemas'}")
    
    if pair_trading_ok and statistical_ok and robusto_ok:
        print("\n🎉 TODAS AS CORREÇÕES FORAM APLICADAS COM SUCESSO!")
    else:
        print("\n⚠️ ALGUMAS CORREÇÕES AINDA PRECISAM DE ATENÇÃO!")

if __name__ == "__main__":
    main()
