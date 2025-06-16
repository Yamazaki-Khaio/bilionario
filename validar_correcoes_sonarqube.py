#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para validação das correções específicas do SonarQube

Este script verifica a implementação de:
1. Método compare_cointegration_methods na classe PairTradingAdvanced
2. Método petrobras_extreme_analysis na classe StatisticalAnalysis
3. Referências ao Prof. Carlos Alberto Rodrigues
"""
import os
import sys
import importlib
import inspect
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

def validate_pair_trading_advanced():
    """Valida implementações na classe PairTradingAdvanced"""
    print_separator()
    print("Validando implementações na classe PairTradingAdvanced")
    print_separator()
    
    try:
        # Abrir e verificar o arquivo de texto diretamente
        with open('pair_trading_advanced.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        print_success("Arquivo pair_trading_advanced.py aberto com sucesso")
            
        # Verificar se os métodos estão definidos no arquivo
        methods = {
            'test_advanced_cointegration': False,
            'compare_cointegration_methods': False
        }
        
        # Para test_advanced_cointegration, verificar as linhas 580-650 onde ele está implementado
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 580 <= i <= 650 and "def test_advanced_cointegration" in line:
                methods['test_advanced_cointegration'] = True
                print_success(f"Método 'test_advanced_cointegration' implementado na classe PairTradingAdvanced (linha {i+1})")
                
                # Verificar documentação nas próximas linhas
                for j in range(i+1, i+10):  # Verificar até 10 linhas depois
                    if j < len(lines) and '"""' in lines[j]:
                        print_success(f"Método 'test_advanced_cointegration' possui documentação")
                        break
                else:
                    print_error(f"Método 'test_advanced_cointegration' não possui documentação")
                break
        else:
            print_error(f"Método 'test_advanced_cointegration' não implementado na classe PairTradingAdvanced")
        
        # Para compare_cointegration_methods, verificar de forma padrão
        pattern = "def compare_cointegration_methods\\s*\\("
        if re.search(pattern, content):
            methods['compare_cointegration_methods'] = True
            print_success(f"Método 'compare_cointegration_methods' implementado na classe PairTradingAdvanced")
            
            # Verificar documentação
            doc_pattern = "def compare_cointegration_methods\\s*\\(.*?\\):\\s*\\n\\s*\"\"\".*?\"\"\""
            if re.search(doc_pattern, content, re.DOTALL):
                print_success(f"Método 'compare_cointegration_methods' possui documentação")
            else:
                print_error(f"Método 'compare_cointegration_methods' não possui documentação")
        else:
            print_error(f"Método 'compare_cointegration_methods' não implementado na classe PairTradingAdvanced")
        
        return all(methods.values())
    except ImportError:
        print_error("Não foi possível importar a classe PairTradingAdvanced")
        return False
    except Exception as e:
        print_error(f"Erro ao validar PairTradingAdvanced: {str(e)}")
        return False

def validate_statistical_analysis():
    """Valida implementações na classe StatisticalAnalysis"""
    print_separator()
    print("Validando implementações na classe StatisticalAnalysis")
    print_separator()
    
    try:
        # Abrir e verificar o arquivo de texto diretamente
        with open('statistical_analysis.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        print_success("Arquivo statistical_analysis.py aberto com sucesso")
        
        # Verificar se o método existe
        method_pattern = r"def petrobras_extreme_analysis\s*\(\s*self\s*,\s*threshold\s*=\s*0\.10\s*\):"
        method_exists = re.search(method_pattern, content) is not None
        
        if method_exists:
            print_success("Método 'petrobras_extreme_analysis' implementado na classe StatisticalAnalysis")
            
            # Verificar documentação
            doc_pattern = r"def petrobras_extreme_analysis\s*\(.*?\):\s*\n\s*\"\"\""
            if re.search(doc_pattern, content):
                print_success("Método 'petrobras_extreme_analysis' possui documentação")
            else:
                print_error("Método 'petrobras_extreme_analysis' não possui documentação")
                
            # Verificar valor padrão do threshold
            threshold_pattern = r"def petrobras_extreme_analysis\s*\(\s*self\s*,\s*threshold\s*=\s*(0\.10)\s*\):"
            match = re.search(threshold_pattern, content)
            
            if match:
                threshold_value = match.group(1)
                if threshold_value == '0.10':
                    print_success("Parâmetro 'threshold' tem valor padrão correto")
                else:
                    print_error(f"Parâmetro 'threshold' tem valor padrão incorreto: {threshold_value}")
            else:
                print_error("Não foi possível verificar o valor padrão do parâmetro 'threshold'")
            
            return True
        else:
            print_error("Método 'petrobras_extreme_analysis' não implementado na classe StatisticalAnalysis")
            return False
    except ImportError:
        print_error("Não foi possível importar a classe StatisticalAnalysis")
        return False
    except Exception as e:
        print_error(f"Erro ao validar StatisticalAnalysis: {str(e)}")
        return False

def validate_carlos_alberto_reference():
    """Valida a referência ao Prof. Carlos Alberto Rodrigues"""
    print_separator()
    print("Validando referência ao Prof. Carlos Alberto Rodrigues")
    print_separator()
    
    files_to_check = [
        "README.md",
        "statistical_analysis.py",
        "pair_trading_advanced.py"
    ]
    
    reference_found = False
    
    for file_path in files_to_check:
        try:
            if not os.path.exists(file_path):
                print(f"Arquivo {file_path} não encontrado")
                continue
                
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            if "Carlos Alberto Rodrigues" in content:
                print_success(f"Referência ao Prof. Carlos Alberto Rodrigues encontrada em {file_path}")
                reference_found = True
        except Exception as e:
            print_error(f"Erro ao verificar arquivo {file_path}: {str(e)}")
    
    if not reference_found:
        print_error("Referência ao Prof. Carlos Alberto Rodrigues não encontrada em nenhum arquivo")
        
    return reference_found

def main():
    """Função principal"""
    print("\nValidação de correções específicas do SonarQube")
    
    # Adicionar diretório atual ao path para importação de módulos
    sys.path.insert(0, os.getcwd())
    
    # Validar implementações
    results = []
    
    results.append(validate_pair_trading_advanced())
    results.append(validate_statistical_analysis())
    results.append(validate_carlos_alberto_reference())
    
    # Calcular porcentagem de correções bem-sucedidas
    success_rate = sum(results) / len(results) if results else 0
    print_separator()
    print(f"Taxa de sucesso: {success_rate:.0%} ({sum(results)}/{len(results)} verificações)")
    
    if success_rate == 1.0:
        print_success("Todas as correções foram implementadas com sucesso!")
    else:
        print_error(f"{len(results) - sum(results)} correções ainda precisam ser implementadas")
    
    return success_rate == 1.0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
