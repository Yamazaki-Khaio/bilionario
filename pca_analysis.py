import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def load_and_prepare_data(file_path='data/raw_data.csv'):
    """
    Carrega e prepara os dados para análise PCA.
    
    Args:
        file_path (str): Caminho para o arquivo de dados
        
    Returns:
        tuple: DataFrame original e matriz de retornos
    """
    # Carregar dados
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    
    # Preencher valores faltantes usando forward fill
    df = df.ffill()
    
    # Calcular retornos usando método mais moderno
    returns = df.pct_change(fill_method=None).dropna()
    
    return df, returns

def perform_pca_analysis(returns, n_components=None):
    """
    Realiza análise PCA completa.
    
    Args:
        returns (pd.DataFrame): Matriz de retornos
        n_components (int): Número de componentes (default: número de ativos)
        
    Returns:
        dict: Resultados da análise PCA
    """
    if n_components is None:
        n_components = min(5, returns.shape[1])
    
    # PCA
    pca = PCA(n_components=n_components, random_state=42)
    components = pca.fit_transform(returns)
    
    # Resultados
    results = {
        'pca': pca,
        'components': components,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
        'loadings': pd.DataFrame(
            pca.components_.T,
            columns=[f'PC{i+1}' for i in range(n_components)],
            index=returns.columns
        )
    }
    
    return results

def plot_pca_results(results, save_plots=True):
    """
    Cria visualizações dos resultados PCA.
    
    Args:
        results (dict): Resultados da análise PCA
        save_plots (bool): Se deve salvar os gráficos
    """
    n_components = len(results['explained_variance_ratio'])
    
    # 1. Scree Plot
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.bar(range(1, n_components + 1), results['explained_variance_ratio'])
    plt.xlabel('Componentes Principais')
    plt.ylabel('Variância Explicada')
    plt.title('Scree Plot - Variância Explicada por Componente')
    plt.grid(True, alpha=0.3)
    
    # 2. Variância Cumulativa
    plt.subplot(2, 2, 2)
    plt.plot(range(1, n_components + 1), results['cumulative_variance'], 'o-')
    plt.xlabel('Número de Componentes')
    plt.ylabel('Variância Explicada Cumulativa')
    plt.title('Variância Explicada Cumulativa')
    plt.grid(True, alpha=0.3)
    
    # 3. Loadings do Primeiro Componente
    plt.subplot(2, 2, 3)
    pc1_loadings = results['loadings']['PC1'].sort_values(key=abs, ascending=False)
    plt.barh(range(len(pc1_loadings)), pc1_loadings.values)
    plt.yticks(range(len(pc1_loadings)), pc1_loadings.index)
    plt.xlabel('Loading')
    plt.title('Loadings do Primeiro Componente Principal')
    plt.grid(True, alpha=0.3)
    
    # 4. Scatter dos dois primeiros componentes (se disponível)
    if n_components >= 2:
        plt.subplot(2, 2, 4)
        plt.scatter(results['components'][:, 0], results['components'][:, 1], alpha=0.6)
        plt.xlabel(f"PC1 ({results['explained_variance_ratio'][0]:.1%})")
        plt.ylabel(f"PC2 ({results['explained_variance_ratio'][1]:.1%})")
        plt.title('Scatter Plot: PC1 vs PC2')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('pca_analysis_complete.png', dpi=300, bbox_inches='tight')
        print('Análise PCA salva como pca_analysis_complete.png')
    
    plt.show()

def plot_correlation_matrix(returns, save_plot=True):
    """
    Plota matriz de correlação dos retornos.
    
    Args:
        returns (pd.DataFrame): Matriz de retornos
        save_plot (bool): Se deve salvar o gráfico
    """
    plt.figure(figsize=(10, 8))
    correlation_matrix = returns.corr()
    
    sns.heatmap(correlation_matrix, 
                annot=True, 
                cmap='coolwarm', 
                center=0,
                square=True,
                fmt='.2f')
    
    plt.title('Matriz de Correlação dos Retornos dos Ativos')
    plt.tight_layout()
    
    if save_plot:
        plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
        print('Matriz de correlação salva como correlation_matrix.png')
    
    plt.show()

def interpret_components(results, threshold=0.3):
    """
    Interpreta os componentes principais baseado nos loadings.
    
    Args:
        results (dict): Resultados da análise PCA
        threshold (float): Limite para considerar loading significativo
        
    Returns:
        dict: Interpretação dos componentes
    """
    interpretations = {}
    
    for i, pc in enumerate(results['loadings'].columns):
        loadings = results['loadings'][pc]
        significant_loadings = loadings[abs(loadings) >= threshold].sort_values(key=abs, ascending=False)
        
        interpretations[pc] = {
            'explained_variance': results['explained_variance_ratio'][i],
            'significant_assets': significant_loadings.to_dict(),
            'interpretation': f"Explica {results['explained_variance_ratio'][i]:.1%} da variância total"
        }
        
        print(f"\n{pc} ({results['explained_variance_ratio'][i]:.1%} da variância):")
        print("Ativos mais significativos:")
        for asset, loading in significant_loadings.items():
            print(f"  {asset}: {loading:.3f}")
    
    return interpretations

def main():
    """Função principal para execução da análise PCA."""
    try:
        # Carregar e preparar dados
        print("Carregando dados...")
        returns = load_and_prepare_data()
        print(f"Dados carregados: {returns.shape[0]} observações, {returns.shape[1]} ativos")
        
        # Análise PCA
        print("\nRealizando análise PCA...")
        results = perform_pca_analysis(returns)
        
        # Matriz de correlação
        print("\nPlotando matriz de correlação...")
        plot_correlation_matrix(returns)
        
        # Visualizações PCA
        print("\nCriando visualizações PCA...")
        plot_pca_results(results)
        
        # Interpretação
        print("\nInterpretando componentes principais...")
        interpretations = interpret_components(results)
        
        # Resumo final
        print("\nResumo da Análise PCA:")
        print(f"Número de componentes analisados: {len(results['explained_variance_ratio'])}")
        print(f"Variância explicada pelos 3 primeiros PCs: {results['cumulative_variance'][2]:.1%}")
        print(f"Primeiro PC explica: {results['explained_variance_ratio'][0]:.1%}")
        
        return results, interpretations
        
    except FileNotFoundError:
        print("Erro: Arquivo 'data/raw_data.csv' não encontrado.")
        print("Execute primeiro o download dos dados na aplicação principal.")
    except Exception as e:
        print(f"Erro durante a análise: {e}")

if __name__ == "__main__":
    results, interpretations = main()
