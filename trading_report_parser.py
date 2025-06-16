"""
Parser especializado para relatórios de trading com dados JavaScript embarcados
Criado para processar relatórios com window.__report
"""

import json
import re
import streamlit as st
import plotly.express as px
from bs4 import BeautifulSoup
from datetime import datetime
import pandas as pd

class TradingReportParser:
    def __init__(self, file_content):
        self.file_content = file_content
        self.report_data = None
        self.portfolio_summary = {}

    def parse(self):
        """Parse principal do relatório de trading"""
        try:
            # Lê o conteúdo HTML
            if hasattr(self.file_content, 'read'):
                raw_content = self.file_content.read()
                
                # Tentar diferentes encodings
                encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252', 'iso-8859-1']
                html_content = None
                
                for encoding in encodings:
                    try:
                        html_content = raw_content.decode(encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                
                if html_content is None:
                    html_content = raw_content.decode('utf-8', errors='replace')
            else:
                html_content = str(self.file_content)
            
            # Extrair dados JavaScript
            self._extract_javascript_data(html_content)
            
            # Processar dados extraídos
            if self.report_data:
                self._process_report_data()
                return True
            else:
                st.error("❌ Não foi possível extrair dados do relatório")
                return False
                
        except Exception as e:
            st.error(f"❌ Erro ao processar relatório: {str(e)}")
            return False    
    
    def _extract_javascript_data(self, html_content):
        """Extrai dados JavaScript do window.__report"""
        patterns = [
            r'window\.__report\s*=\s*({.*?});',
            r'__report\s*=\s*({.*?});',
            r'window\.report\s*=\s*({.*?});',
            # Padrão mais flexível para capturar variações
            r'window\.[_a-zA-Z]+\s*=\s*({.*?});'
        ]
          # Primeiro tenta encontrar o padrão exato para relatórios Trade report-***.html
        trade_report_patterns = [
            r'<script type="text/javascript">window\.__report =\s*({.*?})\s*</script>', 
            r'<script type="text/javascript">window\.__report =\s*({.*})', 
            r'window\.__report =\s*({.*})</script>'
        ]
        
        for pattern in trade_report_patterns:
            match = re.search(pattern, html_content, re.DOTALL)
            if match:
                try:
                    json_str = match.group(1)
                    # Limpar possíveis caracteres inválidos no final
                    if not json_str.endswith('}'):
                        json_str = json_str[:json_str.rfind('}')+1]
                    
                    self.report_data = json.loads(json_str)
                    st.success("✅ Dados extraídos com sucesso do formato Trade report")
                    return
                except json.JSONDecodeError as e:
                    st.warning(f"⚠️ Tentando formato alternativo após erro: {str(e)}")
                    # Continua com outros padrões
        
        # Tenta os padrões padrão
        for pattern in patterns:
            match = re.search(pattern, html_content, re.DOTALL)
            if match:
                try:
                    self.report_data = json.loads(match.group(1))
                    return
                except json.JSONDecodeError as e:
                    st.warning(f"⚠️ Erro ao decodificar JSON: {str(e)}")
                    continue
        
        # Se ainda não encontrou, tenta uma abordagem mais agressiva
        soup = BeautifulSoup(html_content, 'html.parser')
        scripts = soup.find_all('script')
        
        for script in scripts:
            if script.string and 'window.' in script.string:
                script_content = script.string
                for pattern in patterns:
                    match = re.search(pattern, script_content, re.DOTALL)
                    if match:
                        try:
                            self.report_data = json.loads(match.group(1))
                            return
                        except Exception:
                            # Continue tentando
                            pass
        
        st.warning("⚠️ Padrão window.__report não encontrado no arquivo")

    def _process_report_data(self):
        """Processa os dados extraídos em formato padrão"""
        if not self.report_data:
            return

        try:
            # Informações básicas da conta
            account_info = self.report_data.get('summary', {})
            self.portfolio_summary.update({
                'account_name': account_info.get('accountName', 'N/A'),
                'account_number': account_info.get('accountNumber', 'N/A'),
                'currency': account_info.get('currency', 'BRL'),
                'balance': float(account_info.get('balance', 0)),
                'equity': float(account_info.get('equity', 0)),
                'initial_capital': float(account_info.get('initialCapital', 0)),
                'net_profit': float(account_info.get('netProfit', 0)),
                'gain': account_info.get('gainPercent', '0%'),
                'drawdown': account_info.get('drawdownPercent', '0%'),
                'trading_activity': account_info.get('tradingActivity', '0%')
            })

            # Métricas de performance
            metrics = self.report_data.get('metrics', {})
            self.portfolio_summary.update({
                'total_trades': metrics.get('totalTrades', 0),
                'profit_factor': metrics.get('profitFactor', 0),
                'recovery_factor': metrics.get('recoveryFactor', 0),
                'max_consecutive_wins': metrics.get('maxConsecutiveWins', 0),
                'max_consecutive_losses': metrics.get('maxConsecutiveLosses', 0),
                'win_rate': metrics.get('winRate', 0)
            })

            # Dados de balanço histórico
            balance_data = self.report_data.get('balanceData', {})
            if balance_data:
                self.portfolio_summary['balance_history'] = balance_data

            # Dados de trades por símbolo
            symbol_data = self.report_data.get('symbolData', {})
            if symbol_data:
                self.portfolio_summary['symbol_breakdown'] = symbol_data

            st.success("✅ Relatório de trading processado com sucesso!")

        except Exception as e:
            st.error(f"❌ Erro ao processar dados do relatório: {str(e)}")

    def get_portfolio_summary(self):
        """Retorna resumo do portfólio em formato compatível com MT5Parser"""
        return self.portfolio_summary

    def get_trades_dataframe(self):
        """Retorna DataFrame com dados de trades por símbolo"""
        if not self.report_data or 'symbolData' not in self.report_data:
            return None

        try:
            symbol_data = self.report_data['symbolData']
            trades_list = []

            for symbol, data in symbol_data.items():
                trades_list.append({
                    'Symbol': symbol,
                    'NetProfit': data.get('netProfit', 0),
                    'GrossProfit': data.get('grossProfit', 0),
                    'GrossLoss': data.get('grossLoss', 0),
                    'TotalTrades': data.get('totalTrades', 0),
                    'WinRate': data.get('winRate', 0),
                    'ProfitFactor': data.get('profitFactor', 0),
                    'Volume': data.get('volume', 0),
                    'Lots': data.get('lots', 0)
                })

            return pd.DataFrame(trades_list)

        except Exception as e:
            st.error(f"❌ Erro ao criar DataFrame de trades: {str(e)}")
            return None

    def get_balance_history(self):
        """Retorna histórico de balanço como DataFrame"""
        if not self.report_data or 'balanceData' not in self.report_data:
            return None

        try:
            balance_data = self.report_data['balanceData']
            
            # Converter dados de balanço para DataFrame
            dates = balance_data.get('dates', [])
            balance_values = balance_data.get('balance', [])
            equity_values = balance_data.get('equity', [])

            if len(dates) == len(balance_values) == len(equity_values):
                df = pd.DataFrame({
                    'Date': pd.to_datetime(dates),
                    'Balance': balance_values,
                    'Equity': equity_values
                })
                df.set_index('Date', inplace=True)
                return df
            else:
                st.warning("⚠️ Dados de histórico inconsistentes")
                return None

        except Exception as e:
            st.error(f"❌ Erro ao processar histórico de balanço: {str(e)}")
            return None

    def display_summary_metrics(self):
        """Exibe métricas resumidas do relatório"""
        if not self.portfolio_summary:
            st.warning("⚠️ Nenhum dado de resumo disponível")
            return

        st.subheader("📊 Resumo do Relatório de Trading")
        
        # Primeira linha de métricas - com efeito hover
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            balance = self.portfolio_summary.get('balance', 0)
            st.metric("💰 Saldo Atual", f"R$ {balance:,.2f}", 
                     delta=f"R$ {self.portfolio_summary.get('net_profit', 0):,.2f}")
        
        with col2:
            net_profit = self.portfolio_summary.get('net_profit', 0)
            st.metric("📈 Lucro Líquido", f"R$ {net_profit:,.2f}")
        
        with col3:
            gain = self.portfolio_summary.get('gain', '0%')
            st.metric("📊 Ganho Total", gain)
        
        with col4:
            total_trades = self.portfolio_summary.get('total_trades', 0)
            st.metric("🔄 Total de Trades", f"{total_trades}")

        # Segunda linha de métricas
        col5, col6, col7, col8 = st.columns(4)
        with col5:
            profit_factor = self.portfolio_summary.get('profit_factor', 0)
            # Indicar qualidade do profit factor com cores
            if profit_factor >= 2.0:
                pf_delta = "Excelente"
                delta_color = "normal"
            elif profit_factor >= 1.5:
                pf_delta = "Bom"
                delta_color = "normal"
            else:
                pf_delta = "Abaixo do ideal"
                delta_color = "inverse"
            st.metric("⚡ Profit Factor", f"{profit_factor:.2f}", delta=pf_delta, delta_color=delta_color)
        
        with col6:
            win_rate = self.portfolio_summary.get('win_rate', 0)
            # Indicar qualidade do win rate com cores
            if win_rate >= 60:
                wr_delta = "Alto"
                wr_delta_color = "normal"
            elif win_rate >= 50:
                wr_delta = "Médio"
                wr_delta_color = "normal"
            else:
                wr_delta = "Baixo" 
                wr_delta_color = "inverse"
            st.metric("🎯 Taxa de Acerto", f"{win_rate:.1f}%", delta=wr_delta, delta_color=wr_delta_color)
        
        with col7:
            drawdown = self.portfolio_summary.get('drawdown', '0%')
            # Remover o % para comparação
            try:
                dd_value = float(drawdown.replace('%', '').strip())
                if dd_value > 20:
                    dd_color = "red"
                elif dd_value > 10:
                    dd_color = "orange"
                else:
                    dd_color = "green"
                st.metric("📉 Max Drawdown", drawdown, delta="", delta_color=dd_color)
            except:
                st.metric("📉 Max Drawdown", drawdown)
        
        with col8:
            activity = self.portfolio_summary.get('trading_activity', '0%')
            st.metric("⚙️ Atividade", activity)

    def display_symbol_breakdown(self):
        """Exibe breakdown por símbolo"""
        trades_df = self.get_trades_dataframe()
        
        if trades_df is not None and not trades_df.empty:
            st.subheader("📋 Breakdown por Símbolo")
            
            # Formatar colunas monetárias
            for col in ['NetProfit', 'GrossProfit', 'GrossLoss']:
                if col in trades_df.columns:
                    trades_df[col] = trades_df[col].apply(lambda x: f"R$ {x:,.2f}")
            
            # Formatar percentuais
            if 'WinRate' in trades_df.columns:
                trades_df['WinRate'] = trades_df['WinRate'].apply(lambda x: f"{x:.1f}%")
              # Formatar profit factor
            if 'ProfitFactor' in trades_df.columns:
                trades_df['ProfitFactor'] = trades_df['ProfitFactor'].apply(lambda x: f"{x:.2f}")
            
            # Adicionar estilo à tabela com CSS personalizado
            st.markdown("""
            <style>
            .stDataFrame {
                transition: all 0.3s ease;
            }
            .stDataFrame:hover {
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Adicionando um filtro por símbolo acima da tabela
            if len(trades_df) > 5:
                symbols = trades_df['Symbol'].unique().tolist()
                selected_symbols = st.multiselect('Filtrar por símbolo:', symbols, default=symbols[:5])
                if selected_symbols:
                    filtered_df = trades_df[trades_df['Symbol'].isin(selected_symbols)]
                    st.dataframe(filtered_df, use_container_width=True)
                else:
                    st.dataframe(trades_df, use_container_width=True)
            else:
                st.dataframe(trades_df, use_container_width=True)
                
            # Adicionar um gráfico de barras para visualização rápida
            st.subheader("📊 Visualização por Símbolo")
            
            if 'NetProfit' in trades_df.columns:
                # Converter as strings formatadas de volta para números para o gráfico
                trades_df_numeric = trades_df.copy()
                trades_df_numeric['NetProfit_Value'] = trades_df_numeric['NetProfit'].str.replace('R$ ', '').str.replace(',', '').astype(float)
                
                # Ordenar por lucro líquido
                sorted_df = trades_df_numeric.sort_values(by='NetProfit_Value', ascending=False)
                
                fig = px.bar(
                    sorted_df, 
                    x='Symbol', 
                    y='NetProfit_Value', 
                    title='Lucro Líquido por Símbolo',
                    labels={'NetProfit_Value': 'Lucro Líquido (R$)', 'Symbol': 'Símbolo'},
                    color='NetProfit_Value',
                    color_continuous_scale='RdYlGn',
                    text='NetProfit'
                )
                fig.update_layout(
                    height=500, 
                    hoverlabel=dict(bgcolor="white"),
                    xaxis={'categoryorder':'total descending'}
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Dados de breakdown por símbolo não disponíveis")
