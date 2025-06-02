import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Carregar dados
df = pd.read_csv('data/raw_data.csv', index_col=0, parse_dates=True)

# Preencher valores faltantes
df.fillna(method='ffill', inplace=True)

# Normalize
returns = df.pct_change().dropna()

# PCA
pca = PCA(n_components=5)
components = pca.fit_transform(returns)

# Explicar variância
explained = pca.explained_variance_ratio_
print("Variância explicada por componente:", explained)

# Gráfico
plt.figure(figsize=(8,6))
plt.bar(range(1,6), explained)
plt.xlabel('Componentes')
plt.ylabel('Variância Explicada')
plt.title('PCA dos Retornos dos Ativos')
plt.savefig('pca_variance.png')
print('Gráfico salvo como pca_variance.png')
