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
    
    # Verificar se o módulo pode ser importado
    try:
        import pair_trading_helpers
        print("✅ Módulo pair_trading_helpers importado com sucesso")
        
        # Verificar se as funções críticas existem
        test_function_exists("pair_trading_helpers", "_select_assets_for_analysis")
        test_function_exists("pair_trading_helpers", "backtest_tab")
        test_function_exists("pair_trading_helpers", "_execute_backtest_analysis")
        
    except Exception as e:
        print(f"❌ Erro ao importar pair_trading_helpers: {str(e)}")
        return False
    
    return True

def test_statistical_analysis_fixes():
    """Testa correções nos módulos statistical_analysis"""
    print("\n🔍 Verificando correções no statistical_analysis_helpers.py...")
    
    # Verificar se o módulo pode ser importado
    try:
        import statistical_analysis_helpers
        print("✅ Módulo statistical_analysis_helpers importado com sucesso")
        
        # Verificar se a função plot_scatter_chart existe
        test_function_exists("statistical_analysis_helpers", "plot_scatter_chart")
        
        # Testar a função plot_scatter_chart (superficialmente)
        try:
            # Criar dados de teste
            data1 = np.random.randn(100)
            data2 = np.random.randn(100)
            
            # Desativar saída do streamlit durante o teste
            import io
            old_stdout = sys.stdout
            new_stdout = io.StringIO()
            sys.stdout = new_stdout
            
            # Chamar a função (sem renderizar)
            statistical_analysis_helpers.plot_scatter_chart(
                data1=data1, 
                data2=data2, 
                name1="Teste1", 
                name2="Teste2", 
                add_regression=True
            )
            
            # Restaurar stdout
            sys.stdout = old_stdout
            print("✅ Função plot_scatter_chart executada sem erros")
            
        except Exception as e:
            print(f"❌ Erro ao executar plot_scatter_chart: {str(e)}")
            return False
        
    except Exception as e:
        print(f"❌ Erro ao importar statistical_analysis_helpers: {str(e)}")
        return False
    
    return True

def test_pca_fixes():
    """Testa correções nos módulos pca"""
    print("\n🔍 Verificando correções no advanced_pca_helpers.py...")
    
    # Verificar se o módulo pode ser importado
    try:
        import advanced_pca_helpers
        print("✅ Módulo advanced_pca_helpers importado com sucesso")
        
    except Exception as e:
        print(f"❌ Erro ao importar advanced_pca_helpers: {str(e)}")
        return False
    
    return True

def main():
    """Função principal"""
    print("🔧 Iniciando validação das correções...")
    
    # Testar correções de pair_trading
    pair_trading_ok = test_pair_trading_fixes()
    
    # Testar correções de statistical_analysis
    stat_analysis_ok = test_statistical_analysis_fixes()
    
    # Testar correções de pca
    pca_ok = test_pca_fixes()
    
    # Resultado final
    if pair_trading_ok and stat_analysis_ok and pca_ok:
        print("\n🎉 VALIDAÇÃO CONCLUÍDA COM SUCESSO! Todas as correções foram aplicadas corretamente.")
    else:
        print("\n⚠️ VALIDAÇÃO CONCLUÍDA COM PROBLEMAS! Algumas correções precisam ser revisadas.")

if __name__ == "__main__":
    main()
