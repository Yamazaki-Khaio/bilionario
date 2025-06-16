# constants.py
"""Constantes usadas na aplicação Bilionário"""

import os

# Labels de métricas
TOTAL_RETURN_LABEL = "Retorno Total"
MAX_DRAWDOWN_LABEL = "Max Drawdown"
ANNUAL_RETURN_LABEL = "Retorno Anual"
SHARPE_RATIO_LABEL = "Sharpe Ratio"
CORRELATION_LABEL = "Correlação"
WEIGHT_PERCENTAGE_LABEL = "Peso (%)"
FINAL_CAPITAL_LABEL = "Capital Final"
METRIC_LABEL = "Métrica"

# Labels de páginas
PCA_PERFORMANCE_TITLE = "📊 Performance PCA"
MT5_COMPARISON_TITLE = "⚖️ Comparação MT5"
ADVANCED_PCA_TITLE = "🔬 PCA Avançado"
PAIR_TRADING_TITLE = "🔄 Pair Trading"
STATISTICAL_ANALYSIS_TITLE = "📈 Análise Estatística"
NORMALITY_ANALYSIS_TITLE = "🔍 Análise de Normalidade"

# Labels de colunas PCA
LOADING_PC1_LABEL = "Loading PC1"
LOADING_PC2_LABEL = "Loading PC2"

# Mensagens de erro comuns
DATA_NOT_FOUND_MSG = "❌ Dados não encontrados. Vá para a página Home e baixe os dados primeiro."
SELECT_VALID_PAIR_MSG = "📊 Selecione um par válido na aba 'Análise Detalhada' primeiro."

# Nomes de arquivos
RAW_DATA_FILENAME = 'raw_data.csv'

# Diretório de dados
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')

# Função para obter o caminho do arquivo de dados
def get_raw_data_path():
    return os.path.join(DATA_DIR, RAW_DATA_FILENAME)

# Setores de mercado
TELECOMMUNICATIONS_SECTOR = 'telecomunicações'
MATERIALS_SECTOR = 'materiais'  
TECHNOLOGY_SECTOR = 'tecnologia'

# Labels de portfolio
MT5_REAL_LABEL = 'MT5 Real'
PCA_PORTFOLIO_LABEL = 'PCA Portfolio'

# Labels de colunas
SYMBOL_COLUMN = 'Símbolo'
PL_ABS_COLUMN = 'P&L_Abs'
LOSS_LABEL = 'Prejuízo'
PROFIT_LABEL = 'Lucro'

# Ativos específicos
PETR4_SYMBOL = "PETR4.SA"
VALE3_SYMBOL = "VALE3.SA"
ITUB4_SYMBOL = "ITUB4.SA"

# Labels de análise
NORMALIZED_RETURNS_LABEL = "Retornos Normalizados"
CUMULATIVE_RETURNS_LABEL = "Retornos Cumulativos"
PORTFOLIO_VALUE_LABEL = "Valor do Portfolio"
BENCHMARK_LABEL = "Benchmark"
RESULTS_INTERPRETATION_LABEL = "🎯 Interpretação dos Resultados"

# Labels de trading
ENTRY_SIGNAL_LABEL = "Sinal de Entrada"
EXIT_SIGNAL_LABEL = "Sinal de Saída"
POSITION_LABEL = "Posição"
PROFIT_LOSS_LABEL = "Lucro/Prejuízo"

# Labels de visualização
PRICE_LABEL = "Preço"
VOLUME_LABEL = "Volume"
RETURNS_LABEL = "Retornos"
VOLATILITY_LABEL = "Volatilidade"

# Labels de estatística
MEAN_LABEL = "Média"
STD_DEV_LABEL = "Desvio Padrão"
SKEWNESS_LABEL = "Assimetria"
KURTOSIS_LABEL = "Curtose"
SHAPIRO_WILK_LABEL = "Shapiro-Wilk"
JARQUE_BERA_LABEL = "Jarque-Bera"
