import PyPDF2
import re
import streamlit as st
import json
from bs4 import BeautifulSoup

class MT5ReportParser:
    def __init__(self, file_content, file_type):
        self.file_content = file_content
        self.file_type = file_type.lower()
        self.portfolio_data = {}

    def parse(self):
        try:
            if self.file_type == '.pdf':
                self._parse_pdf()
            elif self.file_type in ['.html', '.htm']:
                self._parse_html()
            else:
                raise ValueError("Formato não suportado. Apenas PDF e HTML são aceitos para relatórios MT5")
        except Exception as e:
            st.error(f"Erro ao processar arquivo MT5: {str(e)}")
        return self.portfolio_data

    def _parse_html(self):
        """Parse HTML report from MT5"""
        try:
            # Lê o conteúdo HTML
            if hasattr(self.file_content, 'read'):
                html_content = self.file_content.read().decode('utf-8')
            else:
                html_content = str(self.file_content)
            
            # Extrai o JSON do JavaScript - padrão mais flexível
            json_patterns = [
                r'window\.__report\s*=\s*({.*?});',
                r'__report\s*=\s*({.*?});',
                r'var\s+report\s*=\s*({.*?});',
                r'const\s+report\s*=\s*({.*?});'
            ]
            
            report_data = None
            for pattern in json_patterns:
                json_match = re.search(pattern, html_content, re.DOTALL)
                if json_match:
                    try:
                        report_data = json.loads(json_match.group(1))
                        break
                    except json.JSONDecodeError:
                        continue
            
            if not report_data:
                # Tentar extrair usando BeautifulSoup
                soup = BeautifulSoup(html_content, 'html.parser')
                script_tags = soup.find_all('script')
                
                for script in script_tags:
                    if script.string and '__report' in script.string:
                        # Tentar extrair JSON do script
                        for pattern in json_patterns:
                            match = re.search(pattern, script.string, re.DOTALL)
                            if match:
                                try:
                                    report_data = json.loads(match.group(1))
                                    break
                                except json.JSONDecodeError:
                                    continue
                        if report_data:
                            break
            
            if not report_data:
                raise ValueError("Não foi possível encontrar dados do relatório no HTML")
            
            # Extrai informações da conta
            account = report_data.get('account', {})
            self.portfolio_data['account_name'] = account.get('name', 'N/A')
            self.portfolio_data['account_number'] = str(account.get('account', 'N/A'))
            self.portfolio_data['currency'] = account.get('currency', 'BRL')
            
            # Extrai métricas de performance
            summary = report_data.get('summary', {})
            gain_value = summary.get('gain', 0)
            self.portfolio_data['gain'] = f"{gain_value * 100:.2f}%" if isinstance(gain_value, (int, float)) else "0%"
            
            activity_value = summary.get('activity', 0)
            self.portfolio_data['trading_activity'] = f"{activity_value * 100:.2f}%" if isinstance(activity_value, (int, float)) else "0%"
            
            # Extrai dados de balanço
            balance = report_data.get('balance', {})
            self.portfolio_data['balance'] = balance.get('balance', 0)
            self.portfolio_data['equity'] = balance.get('equity', 0)
            
            # Calcula lucro líquido
            balance_table = balance.get('table', {})
            self.portfolio_data['net_profit'] = balance_table.get('total', 0)
            
            # Capital inicial - estratégia melhorada
            deposits = summary.get('deposit', [])
            if isinstance(deposits, list) and len(deposits) > 0:
                # Pegar o primeiro depósito
                deposit_amount = deposits[0] if isinstance(deposits[0], (int, float)) else 0
                self.portfolio_data['initial_capital'] = deposit_amount
            else:
                # Fallback: tentar calcular pelo balanço
                current_balance = self.portfolio_data['balance']
                net_profit = self.portfolio_data['net_profit']
                if current_balance > 0 and net_profit != 0:
                    self.portfolio_data['initial_capital'] = current_balance - net_profit
                else:
                    self.portfolio_data['initial_capital'] = current_balance if current_balance > 0 else 10000
            
            # Drawdown
            summary_indicators = report_data.get('summaryIndicators', {})
            drawdown_value = summary_indicators.get('drawdown', 0)
            self.portfolio_data['drawdown'] = f"{drawdown_value * 100:.2f}%" if isinstance(drawdown_value, (int, float)) else "0%"
            
            # Indicadores adicionais
            self.portfolio_data['sharp_ratio'] = summary_indicators.get('sharp_ratio', 0)
            self.portfolio_data['profit_factor'] = summary_indicators.get('profit_factor', 0)
            self.portfolio_data['recovery_factor'] = summary_indicators.get('recovery_factor', 0)
            self.portfolio_data['trades_per_week'] = summary_indicators.get('trades_per_week', 0)
            
            # Lucros e perdas totais
            profit_total = report_data.get('profitTotal', {})
            self.portfolio_data['profit'] = profit_total.get('profit', 0)
            self.portfolio_data['loss'] = profit_total.get('loss', 0)
            
            # Dados de símbolos - melhorado
            symbols_total = report_data.get('symbolsTotal', {}).get('total', [])
            symbols = {}
            
            if symbols_total:
                for symbol_data in symbols_total:
                    if isinstance(symbol_data, list) and len(symbol_data) >= 2:
                        symbol_name = symbol_data[0]
                        symbol_profit = symbol_data[1]
                        
                        # Filtrar apenas símbolos válidos
                        if (isinstance(symbol_name, str) and 
                            len(symbol_name) >= 3 and 
                            not symbol_name.isdigit() and
                            any(c.isalpha() for c in symbol_name) and
                            symbol_name not in ['USD', 'BRL', 'EUR', 'GBP']):  # Excluir moedas
                            symbols[symbol_name] = symbol_profit
            
            self.portfolio_data['symbols'] = symbols
            
            # Trades long/short
            long_short = report_data.get('longShortTotal', {})
            self.portfolio_data['long_trades'] = long_short.get('long', 0)
            self.portfolio_data['short_trades'] = long_short.get('short', 0)
            self.portfolio_data['total_trades'] = self.portfolio_data['long_trades'] + self.portfolio_data['short_trades']
            
            # Indicadores long/short
            ls_indicators = report_data.get('longShortIndicators', {})
            win_trades = ls_indicators.get('win_trades', [0, 0])
            if isinstance(win_trades, list):
                self.portfolio_data['win_trades'] = sum(win_trades)
            else:
                self.portfolio_data['win_trades'] = win_trades if isinstance(win_trades, (int, float)) else 0
            
            # Win rate
            if self.portfolio_data['total_trades'] > 0:
                self.portfolio_data['win_rate'] = (self.portfolio_data['win_trades'] / self.portfolio_data['total_trades']) * 100
            else:
                self.portfolio_data['win_rate'] = 0
                
        except Exception as e:
            st.error(f"Erro ao processar HTML MT5: {str(e)}")
            raise

    def _parse_pdf(self):
        reader = PyPDF2.PdfReader(self.file_content)
        text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
        # Extração dos diferentes indicadores
        self._extract_account_info(text)
        self._extract_performance(text)
        self._extract_balance(text)
        self._extract_symbols(text)

    def _extract_account_info(self, text):
        # Captura o nome da conta e o número (9 dígitos) baseado no padrão: "NOME 9D DEMO"
        account_match = re.search(r'^(.+?)\s+(\d{9})\s+DEMO', text, re.MULTILINE)
        if account_match:
            self.portfolio_data['account_name'] = account_match.group(1).strip()
            self.portfolio_data['account_number'] = account_match.group(2)
        self.portfolio_data['currency'] = "BRL"

    def _extract_performance(self, text):
        try:
            # Extração do ganho: procura pela etiqueta "Gain" e captura o valor percentual na linha seguinte
            gain_match = re.search(r'Gain\s*[\r\n]+\s*([+\-]?\d+[.,]\d+)%', text)
            if gain_match:
                self.portfolio_data['gain'] = f"{gain_match.group(1)}%"
            
            # Extração da atividade de trading: captura o valor numérico logo após "Trading Activity"
            ta_match = re.search(r'Trading Activity\s*[\r\n]+\s*([\d.,]+)', text)
            if ta_match:
                self.portfolio_data['trading_activity'] = f"{ta_match.group(1)}%"
            
            # Extração do lucro líquido utilizando "Netto P/L:" e tratando separadores de milhar
            netto_match = re.search(r'Netto P/L:\s*([+\-]?\d+(?:\s\d{3})*[.,]\d+)', text)
            if netto_match:
                profit = float(netto_match.group(1).replace(" ", "").replace(",", "."))
                self.portfolio_data['net_profit'] = profit
        except Exception as e:
            st.warning(f"Erro ao extrair performance: {e}")

    def _extract_balance(self, text):
        try:
            # O valor do Balance pode estar em uma linha abaixo da etiqueta "Balance"
            balance_match = re.search(r'Balance\s*[\r\n]+\s*([\d\s.,]+)', text)
            if balance_match:
                balance = float(balance_match.group(1).replace(" ", "").replace(",", "."))
                self.portfolio_data['balance'] = balance
                # Supondo que o capital inicial seja o balanço atual menos o lucro líquido (se disponível)
                self.portfolio_data['initial_capital'] = balance - self.portfolio_data.get("net_profit", 0)
            
            # Extração do drawdown: procura por um número percentual imediatamente anterior à etiqueta "Drawdown"
            drawdown_match = re.search(r'([\d.,]+)%\s+Drawdown', text)
            if drawdown_match:
                self.portfolio_data['drawdown'] = f"{drawdown_match.group(1)}%"
        except Exception as e:
            st.warning(f"Erro ao extrair balanço: {e}")

    def _extract_symbols(self, text):
        try:
            # Localiza a seção entre "4. Symbols" e "5. Risks"
            symbol_section_match = re.search(r'4\. Symbols(.*?)5\. Risks', text, re.DOTALL)
            if not symbol_section_match:
                return
            content = symbol_section_match.group(1).strip()
            symbols = {}

            # Primeiro: captura tokens combinados, no formato "SIMBOLO-VALOR", ex: PETR4-100.60
            combined_pattern = re.compile(r'([A-Z0-9]+)-([+\-]?\d+(?:\s\d{3})*[.,]\d+)', re.MULTILINE)
            for sym, val in combined_pattern.findall(content):
                try:
                    value = float(val.replace(" ", "").replace(",", "."))
                    symbols[sym] = value
                except:
                    symbols[sym] = None
                # Remove o token já capturado para evitar duplicidade
                content = content.replace(f"{sym}-{val}", "")
            
            # Em seguida, captura pares em que o valor e o símbolo aparecem separadamente.
            # Essa expressão aceita números com separadores de milhar (espaço) e símbolo composto por letras e dígitos.
            separate_pattern = re.compile(r'([+\-]?\d+(?:\s\d{3})*[.,]\d+)\s+([A-Z0-9]+)', re.MULTILINE)
            for val, sym in separate_pattern.findall(content):
                if sym not in symbols:
                    try:
                        value = float(val.replace(" ", "").replace(",", "."))
                        symbols[sym] = value
                    except:
                        symbols[sym] = None
            
            self.portfolio_data['symbols'] = symbols
        except Exception as e:
            st.warning(f"Erro ao extrair símbolos: {e}")

    def get_portfolio_summary(self):
        self.parse()
        
        return {
            'account_name': self.portfolio_data.get('account_name', 'N/A'),
            'account_number': self.portfolio_data.get('account_number', 'N/A'),
            'currency': self.portfolio_data.get('currency', 'BRL'),
            'gain': self.portfolio_data.get('gain', '0%'),
            'trading_activity': self.portfolio_data.get('trading_activity', '0%'),
            'net_profit': self.portfolio_data.get('net_profit', 0),
            'balance': self.portfolio_data.get('balance', 0),
            'equity': self.portfolio_data.get('equity', 0),
            'initial_capital': self.portfolio_data.get('initial_capital', 0),
            'drawdown': self.portfolio_data.get('drawdown', '0%'),
            'profit': self.portfolio_data.get('profit', 0),
            'loss': self.portfolio_data.get('loss', 0),
            'sharp_ratio': self.portfolio_data.get('sharp_ratio', 0),
            'profit_factor': self.portfolio_data.get('profit_factor', 0),
            'recovery_factor': self.portfolio_data.get('recovery_factor', 0),
            'total_trades': self.portfolio_data.get('total_trades', 0),
            'win_trades': self.portfolio_data.get('win_trades', 0),
            'win_rate': self.portfolio_data.get('win_rate', 0),
            'long_trades': self.portfolio_data.get('long_trades', 0),
            'short_trades': self.portfolio_data.get('short_trades', 0),
            'symbols': self.portfolio_data.get('symbols', {})
        }
