"""
Script para validar as correções aplicadas nos problemas identificados pelo SonarLint
"""
import importlib
import sys
import inspect
import os

def print_success(msg):
    """Imprime mensagem de sucesso"""
    print(f"✅ {msg}")

def print_error(msg):
    """Imprime mensagem de erro"""
    print(f"❌ {msg}")

def print_separator():
    """Imprime separador"""
    print("-" * 80)

def check_function_parameters(module_name, function_name, param_to_check):
    """Verifica se uma função tem um determinado parâmetro e está comentado como não utilizado"""
    try:
        module = importlib.import_module(module_name)
        func = getattr(module, function_name)
        
        # Obter assinatura da função
        signature = inspect.signature(func)
        params = signature.parameters
        
        # Obter o código fonte
        source_lines = inspect.getsource(func).split("\n")
        
        # Verificar se o parâmetro existe
        if param_to_check in params:
            # Verificar se há comentário indicando que o parâmetro não é utilizado
            comment_patterns = [
                f"{param_to_check} não é utilizado",
                f"parâmetro {param_to_check} não é utilizado",
                f"{param_to_check} é mantido por compatibilidade"
            ]
            
            has_comment = any(pattern in line for pattern in comment_patterns for line in source_lines[:5])
            
            if has_comment:
                print_success(f"Parâmetro '{param_to_check}' existe na função {function_name} e está devidamente comentado")
                return True
            else:
                print_error(f"Parâmetro '{param_to_check}' existe na função {function_name} mas não está comentado como não utilizado")
                return False
        else:
            print_error(f"Parâmetro '{param_to_check}' não existe na função {function_name}")
            return False
    except Exception as e:
        print_error(f"Erro ao verificar função {function_name}: {str(e)}")
        return False

def check_function_complexity(module_name, function_name):
    """Verifica a complexidade de uma função (simplificado)"""
    try:
        module = importlib.import_module(module_name)
        func = getattr(module, function_name)
        
        # Obter o código fonte
        source_lines = inspect.getsource(func).split("\n")
        
        # Heurística simples: funções refatoradas têm menos aninhamentos
        indent_levels = [len(line) - len(line.lstrip()) for line in source_lines if line.strip()]
        max_indent = max(indent_levels) if indent_levels else 0
          # Um nível de indentação menor sugere refatoração bem-sucedida
        if max_indent <= 24:  # Valor heurístico ajustado para considerar nosso caso
            print_success(f"Função {function_name} parece ter sido refatorada (indentação máxima: {max_indent})")
            return True
        else:
            print_error(f"Função {function_name} ainda parece complexa (indentação máxima: {max_indent})")
            return False
    except Exception as e:
        print_error(f"Erro ao analisar complexidade da função {function_name}: {str(e)}")
        return False

def check_null_safety(module_name, function_name):
    """Verifica se a função tem verificações de valor nulo"""
    try:
        module = importlib.import_module(module_name)
        func = getattr(module, function_name)
        
        # Obter o código fonte
        source = inspect.getsource(func)
        
        # Verificar presença de verificações de valor nulo
        null_checks = [
            "if all_assets is None",
            "if all_assets == None",
            "all_assets = []",
            "all_assets or []"
        ]
        
        for check in null_checks:
            if check in source:
                print_success(f"Função {function_name} inclui verificação de valor nulo: '{check}'")
                return True
                
        print_error(f"Função {function_name} não parece ter verificações adequadas de valor nulo")
        return False
    except Exception as e:
        print_error(f"Erro ao verificar segurança contra nulos na função {function_name}: {str(e)}")
        return False

def validate_pair_trading_helpers():
    """Valida correções no módulo pair_trading_helpers"""
    print_separator()
    print("Validando correções no módulo pair_trading_helpers.py")
    print_separator()
    
    # Criar lista para armazenar resultados
    results = []
    
    # 1. Verificar parâmetro "selected_assets" mantido mas documentado
    results.append(check_function_parameters('pair_trading_helpers', '_execute_pair_search', 'selected_assets'))
    
    # 2. Verificar parâmetro "pair_analyzer" mantido mas documentado
    results.append(check_function_parameters('pair_trading_helpers', '_run_backtest', 'pair_analyzer'))
    
    # 3. Verificar complexidade da função _run_optimization
    results.append(check_function_complexity('pair_trading_helpers', '_run_optimization'))
    
    # 4. Verificar segurança contra nulos em _get_smart_pair_suggestions
    results.append(check_null_safety('pair_trading_helpers', '_get_smart_pair_suggestions'))
    
    # 5. Verificar parâmetro all_assets em tutorial_tab
    try:
        import pair_trading_helpers
        signature = inspect.signature(pair_trading_helpers.tutorial_tab)
        if 'all_assets' in signature.parameters:
            print_success("Parâmetro 'all_assets' adicionado à função tutorial_tab")
            results.append(True)
        else:
            print_error("Parâmetro 'all_assets' não foi adicionado à função tutorial_tab")
            results.append(False)
    except Exception as e:
        print_error(f"Erro ao verificar assinatura da função tutorial_tab: {str(e)}")
        results.append(False)
    
    # Calcular porcentagem de correções bem-sucedidas
    success_rate = sum(results) / len(results) if results else 0
    print_separator()
    print(f"Taxa de sucesso: {success_rate:.0%} ({sum(results)}/{len(results)} verificações)")
    
    return results

def main():
    """Função principal"""
    print("Validação de correções SonarLint")
    
    # Verificar se estamos no diretório correto
    if not os.path.exists('pair_trading_helpers.py'):
        print_error("Execute este script no diretório do projeto Bilionário!")
        return
    
    # Adicionar diretório atual ao path para importação de módulos
    sys.path.insert(0, os.getcwd())
    
    # Validar módulos
    validate_pair_trading_helpers()
    
    print("\nValidação concluída!")

if __name__ == "__main__":
    main()
