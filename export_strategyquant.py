import pandas as pd

def export_to_excel():
    df = pd.read_csv('data/raw_data.csv', index_col=0, parse_dates=True)
    df.fillna(method='ffill', inplace=True)
    df.to_excel('strategyquant_data.xlsx', engine='openpyxl')
    print('Dados exportados para strategyquant_data.xlsx')

if __name__ == '__main__':
    export_to_excel()
