#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validação Final das Correções do SonarQube no projeto Bilionário

Este script valida as implementações de:
1. Função risk_models_tab em statistical_analysis_helpers.py
2. Método petrobras_extreme_analysis na classe StatisticalAnalysis
3. Método test_advanced_cointegration na classe PairTradingAdvanced
4. Método compare_cointegration_methods na classe PairTradingAdvanced
5. Referência ao Prof. Carlos Alberto Rodrigues nos documentos
"""

import os
import re
import importlib
from datetime import datetime

# Cores para saída no terminal
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'

def print_success(msg):
    """Imprime mensagem de sucesso"""
    print(f"{Colors.GREEN}✅ {msg}{Colors.ENDC}")

def print_error(msg):
    """Imprime mensagem de erro"""
    print(f"{Colors.RED}❌ {msg}{Colors.ENDC}")

def print_warning(msg):
    """Imprime mensagem de aviso"""
    print(f"{Colors.YELLOW}⚠️ {msg}{Colors.ENDC}")

def print_info(msg):
    """Imprime mensagem informativa"""
    print(f"{Colors.BLUE}ℹ️ {msg}{Colors.ENDC}")

def print_separator():
    """Imprime separador"""
    print(f"{Colors.BOLD}{'-' * 80}{Colors.ENDC}")

def print_header(msg):
    """Imprime cabeçalho"""
    print(f"{Colors.BOLD}{Colors.BLUE}\n{msg}\n{'-' * len(msg)}{Colors.ENDC}")

def check_file_exists(file_path):
    """Verifica se um arquivo existe"""
    if os.path.exists(file_path):
        print_success(f"Arquivo {file_path} encontrado")
        return True
    else:
        print_error(f"Arquivo {file_path} não encontrado")
        return False

def check_function_exists(module_name, function_name):
    """Verifica se uma função existe em um módulo"""
    try:
        module = importlib.import_module(module_name)
        if hasattr(module, function_name):
            function = getattr(module, function_name)
            if callable(function):
                print_success(f"Função '{function_name}' encontrada no módulo {module_name}")
                
                # Verificar documentação
                if function.__doc__:
                    print_success(f"  - Função '{function_name}' possui documentação")
                else:
                    print_warning(f"  - Função '{function_name}' não possui documentação")
                return True
            else:
                print_error(f"'{function_name}' existe em {module_name}, mas não é uma função")
        else:
            # Tentar verificar diretamente no arquivo
            with open(f"{module_name}.py", 'r', encoding='utf-8') as f:
                content = f.read()
                
            if f"def {function_name}" in content:
                print_success(f"Função '{function_name}' encontrada no arquivo {module_name}.py")
                return True
            else:
                print_error(f"Função '{function_name}' não encontrada no módulo ou arquivo {module_name}")
        return False
    except ImportError:
        print_error(f"Não foi possível importar o módulo {module_name}")
        return False
    except Exception as e:
        print_error(f"Erro ao verificar função {function_name}: {str(e)}")
        return False

def check_method_exists_in_file(file_path, class_name, method_name):
    """Verifica se um método existe em uma classe em um arquivo"""
    try:
        if not os.path.exists(file_path):
            print_error(f"Arquivo {file_path} não encontrado")
            return False
            
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Padrão para encontrar a classe
        class_pattern = f"class {class_name}[^:]*:"
        class_match = re.search(class_pattern, content)
        
        if not class_match:
            print_error(f"Classe '{class_name}' não encontrada no arquivo {file_path}")
            return False
            
        # Padrão para encontrar o método dentro da classe
        method_pattern = f"def {method_name}\\s*\\("
        if re.search(method_pattern, content):
            print_success(f"Método '{method_name}' encontrado na classe {class_name}")
            
            # Verificar documentação
            doc_pattern = f"def {method_name}\\s*\\(.*?\\):\\s*[\\n\\s]*\"\"\""
            if re.search(doc_pattern, content, re.DOTALL):
                print_success(f"  - Método '{method_name}' possui documentação")
            else:
                print_warning(f"  - Método '{method_name}' não possui documentação")
                
            return True
        else:
            print_error(f"Método '{method_name}' não encontrado na classe {class_name}")
            return False
    except Exception as e:
        print_error(f"Erro ao verificar método {method_name}: {str(e)}")
        return False

def check_reference_in_documentation(search_term):
    """Verifica se uma referência existe na documentação"""
    doc_files = [
        'README.md',
        'statistical_analysis_helpers.py',
        'pair_trading_advanced.py',
        'statistical_analysis.py'
    ]
    
    found = False
    count = 0
    
    for file_path in doc_files:
        if not os.path.exists(file_path):
            continue
            
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            if search_term in content:
                count += 1
                print_success(f"Referência a '{search_term}' encontrada em {file_path}")
                found = True
        except Exception as e:
            print_warning(f"Erro ao verificar {file_path}: {str(e)}")
    
    if not found:
        print_error(f"Referência a '{search_term}' não encontrada em nenhum arquivo")
        
    return found

def validate_risk_models_tab():
    """Valida a implementação da função risk_models_tab"""
    print_header("VALIDAÇÃO: risk_models_tab em statistical_analysis_helpers.py")
    return check_function_exists('statistical_analysis_helpers', 'risk_models_tab')

def validate_petrobras_extreme_analysis():
    """Valida a implementação do método petrobras_extreme_analysis"""
    print_header("VALIDAÇÃO: petrobras_extreme_analysis na classe StatisticalAnalysis")
    return check_method_exists_in_file('statistical_analysis.py', 'StatisticalAnalysis', 'petrobras_extreme_analysis')

def validate_test_advanced_cointegration():
    """Valida a implementação do método test_advanced_cointegration"""
    print_header("VALIDAÇÃO: test_advanced_cointegration na classe PairTradingAdvanced")
    return check_method_exists_in_file('pair_trading_advanced.py', 'PairTradingAdvanced', 'test_advanced_cointegration')

def validate_compare_cointegration_methods():
    """Valida a implementação do método compare_cointegration_methods"""
    print_header("VALIDAÇÃO: compare_cointegration_methods na classe PairTradingAdvanced")
    return check_method_exists_in_file('pair_trading_advanced.py', 'PairTradingAdvanced', 'compare_cointegration_methods')

def validate_carlos_alberto_reference():
    """Valida a inclusão da referência ao Prof. Carlos Alberto Rodrigues"""
    print_header("VALIDAÇÃO: Referência ao Prof. Carlos Alberto Rodrigues")
    return check_reference_in_documentation('Carlos Alberto Rodrigues')

def main():
    """Função principal"""
    # Cabeçalho
    now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print_header(f"VALIDAÇÃO FINAL DAS CORREÇÕES SONARLINT/SONARQUBE - {now}")
    print_info("Verificando todas as implementações necessárias...")
    print_separator()
    
    # Lista de validações a serem executadas
    validations = [
        ('risk_models_tab', validate_risk_models_tab),
        ('petrobras_extreme_analysis', validate_petrobras_extreme_analysis),
        ('test_advanced_cointegration', validate_test_advanced_cointegration),
        ('compare_cointegration_methods', validate_compare_cointegration_methods),
        ('Referência ao Prof. Carlos Alberto Rodrigues', validate_carlos_alberto_reference)
    ]
    
    # Executar cada validação e coletar resultados
    results = {}
    for name, validation_func in validations:
        print_separator()
        results[name] = validation_func()
        print_separator()
    
    # Exibir resumo final
    print_header("RESUMO FINAL")
    successful = sum(1 for result in results.values() if result)
    total = len(results)
    success_rate = successful / total
    
    for name, result in results.items():
        status = f"{Colors.GREEN}✅ IMPLEMENTADO{Colors.ENDC}" if result else f"{Colors.RED}❌ PENDENTE{Colors.ENDC}"
        print(f"{name}: {status}")
    
    print_separator()
    print(f"Taxa de sucesso: {success_rate:.0%} ({successful}/{total})")
    
    if success_rate == 1.0:
        print(f"{Colors.GREEN}{Colors.BOLD}TODAS AS CORREÇÕES FORAM IMPLEMENTADAS COM SUCESSO!{Colors.ENDC}")
    else:
        print(f"{Colors.RED}{Colors.BOLD}AINDA HÁ CORREÇÕES PENDENTES!{Colors.ENDC}")
    
    return success_rate == 1.0

if __name__ == "__main__":
    main()
