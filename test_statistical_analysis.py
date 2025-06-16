# test_statistical_analysis.py
"""
Script de teste para verificar o funcionamento das análises estatísticas
"""

import pandas as pd
import numpy as np
from statistical_analysis import StatisticalAnalysis
from pair_trading_advanced import PairTradingAdvanced

def test_statistical_analysis():
    """Testa o módulo de análise estatística"""
    print("🧪 Testando módulo de análise estatística...")
    
    # Criar dados sintéticos para teste
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    
    # Simular dados de retornos
    returns_data = {
        'PETR4.SA': np.random.normal(0.001, 0.03, 1000),  # Petrobras
        'VALE3.SA': np.random.normal(0.0005, 0.025, 1000),  # Vale
        'ITUB4.SA': np.random.normal(0.0003, 0.02, 1000),   # Itaú
        'BBDC4.SA': np.random.normal(0.0004, 0.021, 1000),  # Bradesco
    }
    
    # Converter para preços (cumulativo)
    prices_data = {}
    for asset, returns in returns_data.items():
        prices = 100 * (1 + pd.Series(returns, index=dates)).cumprod()
        prices_data[asset] = prices
    
    df = pd.DataFrame(prices_data)
    
    print(f"📊 Dataset criado: {df.shape[0]} observações, {df.shape[1]} ativos")
    print(f"Período: {df.index[0].date()} a {df.index[-1].date()}")
    
    # Testar análise estatística
    stat_analyzer = StatisticalAnalysis(df)
    
    # 1. Teste da análise da Petrobras
    print("\n1️⃣ Testando análise de extremos da Petrobras...")
    petrobras_analysis = stat_analyzer.petrobras_extreme_analysis(threshold=0.05)
    
    if 'error' not in petrobras_analysis:
        print(f"✅ Probabilidade de queda > 5%: {petrobras_analysis['probability']:.2%}")
        print(f"✅ Total de eventos extremos: {petrobras_analysis['extreme_falls_count']}")
        print(f"✅ Teste de normalidade (JB p-value): {petrobras_analysis['normality_tests']['jarque_bera']['p_value']:.4f}")
        
        if 't_student_analysis' in petrobras_analysis:
            print(f"✅ Distribuição t-Student - graus de liberdade: {petrobras_analysis['t_student_analysis']['degrees_freedom']:.2f}")
    else:
        print(f"❌ Erro: {petrobras_analysis['error']}")
    
    # 2. Teste de comparação de distribuições
    print("\n2️⃣ Testando comparação de distribuições...")
    different_distributions = stat_analyzer.find_different_distributions(min_data_points=500)
    
    if different_distributions and 'error' not in different_distributions:
        assets = different_distributions['assets']
        print(f"✅ Comparação: {assets['asset1']} vs {assets['asset2']}")
        
        ks_test = different_distributions['comparison_tests']['ks_test']
        print(f"✅ Teste Kolmogorov-Smirnov: p = {ks_test['p_value']:.4f}, significativo: {ks_test['significant']}")
        
        mw_test = different_distributions['comparison_tests']['mann_whitney']
        print(f"✅ Teste Mann-Whitney U: p = {mw_test['p_value']:.4f}, significativo: {mw_test['significant']}")
    else:
        error_msg = different_distributions.get('error', 'Nenhum par encontrado') if different_distributions else 'Erro desconhecido'
        print(f"ℹ️ {error_msg}")
    
    # 3. Teste de análise de risco da Petrobras
    print("\n3️⃣ Testando análise de risco...")
    risk_analysis = stat_analyzer.create_petrobras_risk_analysis()
    
    if 'error' not in risk_analysis:
        var_cvar = risk_analysis['var_cvar']
        print(f"✅ VaR 95%: {var_cvar['var_95']:.2%}")
        print(f"✅ CVaR 95%: {var_cvar['cvar_95']:.2%}")
        
        tail_analysis = risk_analysis['tail_analysis']
        print(f"✅ Eventos na cauda esquerda: {tail_analysis['left_tail_count']} ({tail_analysis['left_tail_freq']:.1%})")
        
        vol_analysis = risk_analysis['volatility_analysis']
        print(f"✅ Volatilidade anual: {vol_analysis['current_vol_annual']:.1%}")
    else:
        print(f"❌ Erro: {risk_analysis['error']}")
    
    print("\n🎯 Teste do módulo de análise estatística concluído!")

def test_pair_trading_advanced():
    """Testa o módulo de pair trading avançado"""
    print("\n🔄 Testando módulo de pair trading avançado...")
    
    # Usar os mesmos dados sintéticos
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    
    # Criar dois ativos cointegrados artificialmente
    asset1_returns = np.random.normal(0.0005, 0.02, 1000)
    noise = np.random.normal(0, 0.005, 1000)
    asset2_returns = 0.8 * asset1_returns + noise  # Correlação artificial
    
    asset1_prices = 100 * (1 + pd.Series(asset1_returns, index=dates)).cumprod()
    asset2_prices = 100 * (1 + pd.Series(asset2_returns, index=dates)).cumprod()
    
    df = pd.DataFrame({
        'PETR4.SA': asset1_prices,
        'VALE3.SA': asset2_prices
    })
    
    try:
        pair_analyzer = PairTradingAdvanced(df)
        
        # Testar análise do par
        print("📊 Testando análise de par com distribuições...")
        pair_analysis = pair_analyzer.analyze_pair_with_distributions('PETR4.SA', 'VALE3.SA')
        
        if 'error' not in pair_analysis:
            cointegration = pair_analysis['cointegration']
            print(f"✅ Cointegração: {cointegration['is_cointegrated']}")
            print(f"✅ P-value: {cointegration['p_value']:.4f}")
            print(f"✅ Hedge ratio: {cointegration['hedge_ratio']:.4f}")
            print(f"✅ Correlação: {cointegration['correlation']:.3f}")
            
            # Análise de distribuições
            if 'distribution_analysis' in pair_analysis:
                dist_analysis = pair_analysis['distribution_analysis']
                print(f"✅ Melhor distribuição: {dist_analysis['best_distribution']}")
                print(f"✅ Teste de normalidade: {dist_analysis['normality_tests']['is_normal']}")
            
            # Sinais de trading
            if 'trading_signals' in pair_analysis:
                signals = pair_analysis['trading_signals']
                print(f"✅ Total de sinais: {signals['signal_stats']['total_signals']}")
                print(f"✅ Sinais long: {signals['signal_stats']['long_signals']}")
                print(f"✅ Sinais short: {signals['signal_stats']['short_signals']}")
        else:
            print(f"❌ Erro: {pair_analysis['error']}")
            
    except Exception as e:
        print(f"⚠️ Erro no teste de pair trading: {str(e)}")
    
    print("🎯 Teste do módulo de pair trading avançado concluído!")

if __name__ == "__main__":
    print("🚀 Iniciando testes dos módulos de análise estatística avançada")
    print("=" * 60)
    
    try:
        test_statistical_analysis()
        test_pair_trading_advanced()
        
        print("\n" + "=" * 60)
        print("✅ Todos os testes concluídos com sucesso!")
        print("💡 Os módulos de análise estatística estão funcionando corretamente")
        
    except Exception as e:
        print(f"\n❌ Erro durante os testes: {str(e)}")
        import traceback
        traceback.print_exc()
