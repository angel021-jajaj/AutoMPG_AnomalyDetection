import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

# 1. Cargar y preprocesar el conjunto de datos Auto MPG
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
column_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin']
data = pd.read_csv(url, names=column_names, na_values='?', comment='\t', sep=' ', skipinitialspace=True)

# Manejar valores faltantes
data = data.dropna()

# Separar características (excluimos 'mpg')
X = data.drop('mpg', axis=1).values

# Escalar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Detectar anomalías con k-NN
k = 5  # Número de vecinos
nbrs = NearestNeighbors(n_neighbors=k).fit(X_scaled)
distances, _ = nbrs.kneighbors(X_scaled)

# Usar la distancia al k-ésimo vecino
kth_distances = distances[:, -1]

# Establecer un umbral para anomalías (percentil 95)
threshold_knn = np.percentile(kth_distances, 95)
anomalies_knn = kth_distances > threshold_knn

print(f"Anomalías detectadas por k-NN: {np.sum(anomalies_knn)}")