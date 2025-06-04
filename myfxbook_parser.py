import pandas as pd
import PyPDF2
import re
from bs4 import BeautifulSoup
from pathlib import Path
import streamlit as st

class MyfxbookParser:
    """
    Parser para extrair dados de relatórios Myfxbook em diferentes formatos.
    Suporta PDF, CSV e HTML.
    """
    
    def __init__(self, file_content, file_type):
        """
        Inicializa o parser com o conteúdo do arquivo.
        
        Args:
            file_content: Conteúdo do arquivo carregado
            file_type (str): Tipo do arquivo (.pdf, .csv, .html)
        """
        self.file_content = file_content
        self.file_type = file_type.lower()
        self.portfolio_data = {}
        
    def parse(self):
        """
        Analisa o arquivo de acordo com seu formato e extrai os dados.
        
        Returns:
            dict: Dados do portfólio extraídos
        """
        try:
            if self.file_type == '.pdf':
                self._parse_pdf()
            elif self.file_type == '.csv':
                self._parse_csv()
            elif self.file_type in ['.html', '.htm']:
                self._parse_html()
            else:
                raise ValueError(f"Formato não suportado: {self.file_type}")
        except Exception as e:
            st.error(f"Erro ao processar arquivo: {str(e)}")
            return {}
        
        return self.portfolio_data
    
    def _parse_pdf(self):
        """Extrai dados do relatório em formato PDF."""
        try:
            # Para PDF, tentamos extrair texto
            reader = PyPDF2.PdfReader(self.file_content)
            text = ""
            
            for page in reader.pages:
                text += page.extract_text()
            
            self._extract_data_from_text(text)
        except Exception as e:
            st.error(f"Erro ao processar PDF: {e}")
    
    def _parse_csv(self):
        """Extrai dados do relatório em formato CSV."""
        try:
            # Converte bytes para string se necessário
            if isinstance(self.file_content, bytes):
                content = self.file_content.decode('utf-8')
            else:
                content = self.file_content
            
            # Cria DataFrame a partir do conteúdo CSV
            from io import StringIO
            df = pd.read_csv(StringIO(content))
            
            self._extract_data_from_dataframe(df)
        except Exception as e:
            st.error(f"Erro ao processar CSV: {e}")
    
    def _parse_html(self):
        """Extrai dados do relatório em formato HTML."""
        try:
            # Converte bytes para string se necessário
            if isinstance(self.file_content, bytes):
                content = self.file_content.decode('utf-8')
            else:
                content = self.file_content
            
            soup = BeautifulSoup(content, 'html.parser')
            self._extract_data_from_html(soup)
        except Exception as e:
            st.error(f"Erro ao processar HTML: {e}")
    
    def _extract_data_from_text(self, text):
        """
        Extrai informações do texto do relatório PDF.
        
        Args:
            text (str): Texto extraído do PDF
        """
        # Padrões de extração para relatórios Myfxbook
        patterns = {
            'account_name': r'Account Name[:\s]+([^\n\r]+)',
            'balance': r'Balance[:\s]+\$?([0-9,.-]+)',
            'equity': r'Equity[:\s]+\$?([0-9,.-]+)',
            'profit': r'Profit[:\s]+\$?([0-9,.-]+)',
            'gain': r'Gain[:\s]+([0-9,.-]+)%?',
            'drawdown': r'Drawdown[:\s]+([0-9,.-]+)%?',
            'trading_period': r'Trading[:\s]+([0-9]+ days?)',
            'trades': r'Trades[:\s]+([0-9,]+)',
            'pips': r'Pips[:\s]+([0-9,.-]+)',
            'win_rate': r'Win Rate[:\s]+([0-9,.-]+)%?'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                # Limpa e converte valores numéricos
                if key in ['balance', 'equity', 'profit', 'gain', 'drawdown', 'pips', 'win_rate']:
                    cleaned_value = self._clean_numeric_value(value)
                    self.portfolio_data[key] = cleaned_value
                else:
                    self.portfolio_data[key] = value
    
    def _extract_data_from_dataframe(self, df):
        """
        Extrai informações do DataFrame carregado do CSV.
        
        Args:
            df (pandas.DataFrame): DataFrame com dados do relatório
        """
        # Mapeia colunas comuns de CSV do Myfxbook
        column_mapping = {
            'Account Name': 'account_name',
            'Balance': 'balance',
            'Equity': 'equity',
            'Profit': 'profit',
            'Gain': 'gain',
            'Drawdown': 'drawdown',
            'Trades': 'trades',
            'Pips': 'pips',
            'Win Rate': 'win_rate'
        }
        
        for col, key in column_mapping.items():
            if col in df.columns:
                value = df[col].iloc[0] if len(df) > 0 else None
                if value is not None:
                    if key in ['balance', 'equity', 'profit', 'gain', 'drawdown', 'pips', 'win_rate']:
                        self.portfolio_data[key] = self._clean_numeric_value(str(value))
                    else:
                        self.portfolio_data[key] = value
        
        # Se há dados de trades individuais
        trade_columns = ['Open Time', 'Close Time', 'Symbol', 'Action', 'Lots', 'Profit']
        if all(col in df.columns for col in trade_columns):
            self.portfolio_data['trade_history'] = df[trade_columns].to_dict('records')
    
    def _extract_data_from_html(self, soup):
        """
        Extrai informações da estrutura HTML.
        
        Args:
            soup (BeautifulSoup): Objeto BeautifulSoup com o HTML do relatório
        """
        # Extrai título da página
        title = soup.find('title')
        if title:
            self.portfolio_data['account_name'] = title.text.split('|')[0].strip()
        
        # Procura por tabelas com métricas
        tables = soup.find_all('table')
        for table in tables:
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 2:
                    label = cells[0].text.strip().lower()
                    value = cells[1].text.strip()
                    
                    # Mapeia labels para chaves
                    if 'balance' in label:
                        self.portfolio_data['balance'] = self._clean_numeric_value(value)
                    elif 'equity' in label:
                        self.portfolio_data['equity'] = self._clean_numeric_value(value)
                    elif 'profit' in label:
                        self.portfolio_data['profit'] = self._clean_numeric_value(value)
                    elif 'gain' in label:
                        self.portfolio_data['gain'] = self._clean_numeric_value(value)
                    elif 'drawdown' in label:
                        self.portfolio_data['drawdown'] = self._clean_numeric_value(value)
    
    def _clean_numeric_value(self, value):
        """
        Limpa e converte valores numéricos de forma segura.
        
        Args:
            value (str): Valor a ser limpo
            
        Returns:
            float: Valor numérico limpo ou 0.0 se não conseguir converter
        """
        try:
            # Remove símbolos e espaços
            cleaned = str(value).replace('$', '').replace('%', '').replace(',', '').replace(' ', '')
            # Tenta converter para float
            return float(cleaned) if cleaned and cleaned != 'N/A' else 0.0
        except (ValueError, TypeError, AttributeError):
            return 0.0
    
    def get_portfolio_summary(self):
        """
        Retorna um resumo do portfólio extraído.
        
        Returns:
            dict: Resumo do portfólio com valores garantidamente numéricos
        """
        if not self.portfolio_data:
            self.parse()
        
        return {
            'account_name': self.portfolio_data.get('account_name', 'N/A'),
            'balance': self._clean_numeric_value(self.portfolio_data.get('balance', 0)),
            'equity': self._clean_numeric_value(self.portfolio_data.get('equity', 0)),
            'profit': self._clean_numeric_value(self.portfolio_data.get('profit', 0)),
            'gain_percent': self._clean_numeric_value(self.portfolio_data.get('gain', 0)),
            'drawdown_percent': self._clean_numeric_value(self.portfolio_data.get('drawdown', 0)),
            'total_trades': int(self._clean_numeric_value(self.portfolio_data.get('trades', 0))),
            'total_pips': self._clean_numeric_value(self.portfolio_data.get('pips', 0)),
            'win_rate': self._clean_numeric_value(self.portfolio_data.get('win_rate', 0))
        }