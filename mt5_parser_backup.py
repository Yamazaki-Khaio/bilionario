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
        """Parse HTML report from MT5 with enhanced pattern matching"""
        try:
            # Lê o conteúdo HTML com tratamento de encoding
            if hasattr(self.file_content, 'read'):
                # Tentar diferentes encodings para evitar erro UTF-8
                raw_content = self.file_content.read()
                
                # Lista de encodings para tentar
                encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252', 'iso-8859-1']
                
                html_content = None
                for encoding in encodings:
                    try:
                        html_content = raw_content.decode(encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                
                if html_content is None:
                    # Se todos os encodings falharam, usar errors='replace'
                    html_content = raw_content.decode('utf-8', errors='replace')
            else:
                html_content = str(self.file_content)
            
            # Debug do conteúdo HTML (opcional)
            # self.debug_html_content(html_content)
            
            # Múltiplas estratégias de parsing
            parsed_data = None
            
            # Estratégia 1: JSON patterns (original)
            parsed_data = self._try_json_extraction(html_content)
            
            # Estratégia 2: HTML table parsing
            if not parsed_data:
                parsed_data = self._try_html_table_parsing(html_content)
            
            # Estratégia 3: Regex patterns para formatos específicos
            if not parsed_data:
                parsed_data = self._try_regex_patterns(html_content)
            
            # Estratégia 4: BeautifulSoup com múltiplos seletores
            if not parsed_data:
                parsed_data = self._try_beautifulsoup_parsing(html_content)
            
            if not parsed_data:
                # Se nenhuma estratégia funcionou, criar dados padrão
                st.warning("⚠️ Formato de relatório não reconhecido. Usando valores padrão.")
                parsed_data = self._create_default_data()
            
            # Processar dados extraídos
            self._process_extracted_data(parsed_data)
                
        except Exception as e:
            st.error(f"Erro ao processar HTML MT5: {str(e)}")
            # Criar dados padrão em caso de erro
            self.portfolio_data = self._create_default_data()

    def _try_json_extraction(self, html_content):
        """Estratégia 1: Extração de JSON do JavaScript"""
        json_patterns = [
            r'window\.__report\s*=\s*({.*?});',
            r'__report\s*=\s*({.*?});',
            r'var\s+report\s*=\s*({.*?});',
            r'const\s+report\s*=\s*({.*?});',
            r'window\.reportData\s*=\s*({.*?});',
            r'reportData\s*=\s*({.*?});'
        ]
        
        for pattern in json_patterns:
            json_match = re.search(pattern, html_content, re.DOTALL)            
            if json_match:
                try:
                    return json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    continue
        return None

    def _try_html_table_parsing(self, html_content):
        """Estratégia 2: Parsing de tabelas HTML"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            tables = soup.find_all('table')
            
            extracted_data = {}
            for table in tables:
                self._extract_table_data(table, extracted_data)
            
            return extracted_data if extracted_data else None
            
        except Exception:
            return None

    def _extract_table_data(self, table, extracted_data):
        """Extrai dados de uma tabela HTML específica"""
        rows = table.find_all('tr')
        for row in rows:
            cells = row.find_all(['td', 'th'])
            if len(cells) >= 2:
                key = cells[0].get_text(strip=True).lower()
                value = cells[1].get_text(strip=True)
                self._map_table_field(key, value, extracted_data)

    def _map_table_field(self, key, value, extracted_data):
        """Mapeia campo da tabela para dados extraídos"""
        field_mappings = {
            ('balance', 'saldo'): ('balance', self._parse_currency),
            ('profit', 'lucro'): ('net_profit', self._parse_currency),
            ('equity', 'patrimônio'): ('equity', self._parse_currency),
            ('drawdown',): ('drawdown', self._parse_percentage)
        }
        
        for keywords, (field_name, parser_func) in field_mappings.items():
            if any(keyword in key for keyword in keywords):
                extracted_data[field_name] = parser_func(value)
                break

    def _try_regex_patterns(self, html_content):
        """Estratégia 3: Patterns específicos de regex"""
        try:
            patterns = {
                'balance': [
                    r'Balance[:\s]*([R$\d,.-]+)',
                    r'Saldo[:\s]*([R$\d,.-]+)',
                    r'balance["\s]*:["\s]*([R$\d,.-]+)'
                ],
                'profit': [
                    r'Profit[:\s]*([R$\d,.-]+)',
                    r'Lucro[:\s]*([R$\d,.-]+)',
                    r'profit["\s]*:["\s]*([R$\d,.-]+)'
                ],
                'equity': [
                    r'Equity[:\s]*([R$\d,.-]+)',
                    r'Patrimônio[:\s]*([R$\d,.-]+)',
                    r'equity["\s]*:["\s]*([R$\d,.-]+)'
                ]
            }
            
            extracted_data = {}
            
            for field, pattern_list in patterns.items():
                for pattern in pattern_list:
                    match = re.search(pattern, html_content, re.IGNORECASE)
                    if match:
                        extracted_data[field] = self._parse_currency(match.group(1))
                        break
            
            return extracted_data if extracted_data else None
            
        except Exception:
            return None

    def _try_beautifulsoup_parsing(self, html_content):
        """Estratégia 4: BeautifulSoup com seletores específicos"""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Procurar por elementos com classes ou IDs específicos
            selectors = [
                '.balance', '#balance', '[data-balance]',
                '.profit', '#profit', '[data-profit]',
                '.equity', '#equity', '[data-equity]',
                '.account', '#account', '[data-account]'
            ]
            
            extracted_data = {}
            
            for selector in selectors:
                elements = soup.select(selector)
                for element in elements:
                    text = element.get_text(strip=True)
                    if text and any(c.isdigit() for c in text):
                        field_name = selector.replace('.', '').replace('#', '').replace('[data-', '').replace(']', '')
                        extracted_data[field_name] = self._parse_currency(text)
            
            return extracted_data if extracted_data else None
            
        except Exception:
            return None

    def _parse_currency(self, value):
        """Converte string de moeda para float"""
        try:
            # Remove símbolos de moeda e espaços
            clean_value = re.sub(r'[R$\s]', '', str(value))            # Substitui vírgula por ponto se for decimal brasileiro
            if ',' in clean_value and '.' not in clean_value:
                clean_value = clean_value.replace(',', '.')
            elif ',' in clean_value and '.' in clean_value:
                # Formato brasileiro: 1.234,56
                clean_value = clean_value.replace('.', '').replace(',', '.')
            
            return float(clean_value)
        except (ValueError, TypeError):
            return 0.0

    def _parse_percentage(self, value):
        """Converte string de porcentagem para formato decimal"""
        try:
            clean_value = re.sub(r'[%\s]', '', str(value))
            if ',' in clean_value:
                clean_value = clean_value.replace(',', '.')
            percentage = float(clean_value)
            return f"{percentage:.2f}%"
        except (ValueError, TypeError):
            return "0%"

    def _create_default_data(self):
        """Cria dados padrão quando parsing falha"""
        return {
            'account': {'name': 'N/A', 'account': 'N/A', 'currency': 'BRL'},
            'summary': {'gain': 0, 'activity': 0},
            'balance': {'balance': 0, 'equity': 0, 'table': {'total': 0}},
            'summaryIndicators': {
                'drawdown': 0, 'sharp_ratio': 0, 'profit_factor': 0,
                'recovery_factor': 0, 'trades_per_week': 0
            },            'profitTotal': {'profit': 0, 'loss': 0},
            'symbolsTotal': {'total': []},
            'longShortTotal': {'long': 0, 'short': 0},
            'longShortIndicators': {'win_trades': 0}
        }

    def _process_extracted_data(self, report_data):
        """Processa dados extraídos e preenche portfolio_data"""
        # Verificação robusta de report_data
        if not report_data or not isinstance(report_data, dict):
            st.warning("⚠️ Dados do relatório inválidos. Usando valores padrão.")
            report_data = self._create_default_data()
        
        # Processa cada seção de dados
        self._process_account_data(report_data)
        self._process_performance_data(report_data)
        self._process_balance_data(report_data)
        self._process_capital_data(report_data)
        self._process_indicators_data(report_data)
        self._process_profit_loss_data(report_data)
        self._process_symbols_data(report_data)
        self._process_trading_data(report_data)

    def _process_account_data(self, report_data):
        """Processa informações da conta"""
        account = report_data.get('account', {})
        if not isinstance(account, dict):
            account = {}
        
        self.portfolio_data['account_name'] = account.get('name', 'N/A')
        self.portfolio_data['account_number'] = str(account.get('account', 'N/A'))
        self.portfolio_data['currency'] = account.get('currency', 'BRL')

    def _process_performance_data(self, report_data):
        """Processa métricas de performance"""
        summary = report_data.get('summary', {})
        if not isinstance(summary, dict):
            summary = {}
            
        gain_value = summary.get('gain', 0)
        self.portfolio_data['gain'] = f"{gain_value * 100:.2f}%" if isinstance(gain_value, (int, float)) else "0%"
        
        activity_value = summary.get('activity', 0)
        self.portfolio_data['trading_activity'] = f"{activity_value * 100:.2f}%" if isinstance(activity_value, (int, float)) else "0%"

    def _process_balance_data(self, report_data):
        """Processa dados de balanço"""
        balance = report_data.get('balance', {})
        if not isinstance(balance, dict):
            balance = {}
            
        self.portfolio_data['balance'] = balance.get('balance', 0)
        self.portfolio_data['equity'] = balance.get('equity', 0)
        
        # Calcula lucro líquido
        balance_table = balance.get('table', {})
        if not isinstance(balance_table, dict):
            balance_table = {}
            
        self.portfolio_data['net_profit'] = balance_table.get('total', 0)
        
        # Se dados diretos existem, usar eles
        if 'balance' in report_data and isinstance(report_data['balance'], (int, float)):
            self.portfolio_data['balance'] = report_data['balance']
        if 'profit' in report_data and isinstance(report_data['profit'], (int, float)):
            self.portfolio_data['net_profit'] = report_data['profit']
        if 'equity' in report_data and isinstance(report_data['equity'], (int, float)):
            self.portfolio_data['equity'] = report_data['equity']

    def _process_capital_data(self, report_data):
        """Processa dados de capital inicial"""
        summary = report_data.get('summary', {})
        if not isinstance(summary, dict):
            summary = {}
            
        deposits = summary.get('deposit', [])
        if isinstance(deposits, list) and len(deposits) > 0:
            deposit_amount = deposits[0] if isinstance(deposits[0], (int, float)) else 0
            self.portfolio_data['initial_capital'] = deposit_amount
        else:
            # Fallback: calcular pelo balanço
            current_balance = self.portfolio_data.get('balance', 0)
            net_profit = self.portfolio_data.get('net_profit', 0)
            if current_balance > 0 and net_profit != 0:
                self.portfolio_data['initial_capital'] = current_balance - net_profit
            else:
                self.portfolio_data['initial_capital'] = current_balance if current_balance > 0 else 10000

    def _process_indicators_data(self, report_data):
        """Processa indicadores de performance"""
        summary_indicators = report_data.get('summaryIndicators', {})
        
        drawdown_value = summary_indicators.get('drawdown', 0)
        self.portfolio_data['drawdown'] = f"{drawdown_value * 100:.2f}%" if isinstance(drawdown_value, (int, float)) else "0%"
        
        self.portfolio_data['sharp_ratio'] = summary_indicators.get('sharp_ratio', 0)
        self.portfolio_data['profit_factor'] = summary_indicators.get('profit_factor', 0)
        self.portfolio_data['recovery_factor'] = summary_indicators.get('recovery_factor', 0)
        self.portfolio_data['trades_per_week'] = summary_indicators.get('trades_per_week', 0)

    def _process_profit_loss_data(self, report_data):
        """Processa dados de lucros e perdas"""
        profit_total = report_data.get('profitTotal', {})
        self.portfolio_data['profit'] = profit_total.get('profit', 0)
        self.portfolio_data['loss'] = profit_total.get('loss', 0)

    def _process_symbols_data(self, report_data):
        """Processa dados de símbolos"""
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
                        symbol_name not in ['USD', 'BRL', 'EUR', 'GBP']):
                        symbols[symbol_name] = symbol_profit
        
        self.portfolio_data['symbols'] = symbols

    def _process_trading_data(self, report_data):
        """Processa dados de trading"""
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

    def debug_html_content(self, html_content, max_length=1000):
        """Função de debug para analisar conteúdo HTML"""
        st.write("🔍 **Debug - Conteúdo HTML:**")
        
        # Mostrar início do conteúdo
        content_preview = html_content[:max_length]
        if len(html_content) > max_length:
            content_preview += "... (truncado)"
        
        st.code(content_preview, language='html')
        
        # Procurar por padrões conhecidos
        patterns_found = []
        
        search_patterns = [
            ('JSON Report', r'__report|window\.__report|reportData'),
            ('HTML Tables', r'<table|<tr|<td'),
            ('Balance Keywords', r'balance|saldo|Balance'),
            ('Profit Keywords', r'profit|lucro|Profit'),
            ('Script Tags', r'<script'),
            ('MetaTrader', r'MetaTrader|MT5|mt5'),
        ]
        
        for pattern_name, pattern in search_patterns:
            if re.search(pattern, html_content, re.IGNORECASE):
                patterns_found.append(pattern_name)
        
        if patterns_found:
            st.write(f"**Padrões encontrados:** {', '.join(patterns_found)}")
        else:
            st.warning("⚠️ Nenhum padrão conhecido encontrado")
        
        # Verificar tamanho do arquivo
        st.write(f"**Tamanho do arquivo:** {len(html_content)} caracteres")

    def _parse_pdf(self):
        """Parse PDF report from MT5"""
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
        """Extrai informações da conta do texto PDF"""
        # Implementar extração de conta para PDF
        pass

    def _extract_performance(self, text):
        """Extrai métricas de performance do texto PDF"""
        # Implementar extração de performance para PDF
        pass

    def _extract_balance(self, text):
        """Extrai dados de balanço do texto PDF"""
        # Implementar extração de balanço para PDF
        pass

    def _extract_symbols(self, text):
        """Extrai símbolos do texto PDF"""
        # Implementar extração de símbolos para PDF
        pass

    def get_portfolio_summary(self):
        """Retorna resumo do portfólio MT5 processado"""
        if not self.portfolio_data:
            return None
        
        return {
            'account_name': self.portfolio_data.get('account_name', 'N/A'),
            'account_number': self.portfolio_data.get('account_number', 'N/A'),
            'currency': self.portfolio_data.get('currency', 'BRL'),
            'balance': self.portfolio_data.get('balance', 0),
            'equity': self.portfolio_data.get('equity', 0),
            'net_profit': self.portfolio_data.get('net_profit', 0),
            'initial_capital': self.portfolio_data.get('initial_capital', 0),
            'gain': self.portfolio_data.get('gain', '0%'),
            'drawdown': self.portfolio_data.get('drawdown', '0%'),
            'trading_activity': self.portfolio_data.get('trading_activity', '0%'),
            'sharp_ratio': self.portfolio_data.get('sharp_ratio', 0),
            'profit_factor': self.portfolio_data.get('profit_factor', 0),
            'recovery_factor': self.portfolio_data.get('recovery_factor', 0),
            'trades_per_week': self.portfolio_data.get('trades_per_week', 0),
            'profit': self.portfolio_data.get('profit', 0),
            'loss': self.portfolio_data.get('loss', 0),
            'symbols': self.portfolio_data.get('symbols', {}),
            'long_trades': self.portfolio_data.get('long_trades', 0),
            'short_trades': self.portfolio_data.get('short_trades', 0),
            'total_trades': self.portfolio_data.get('total_trades', 0),
            'win_trades': self.portfolio_data.get('win_trades', 0),
            'win_rate': self.portfolio_data.get('win_rate', 0)
        }
