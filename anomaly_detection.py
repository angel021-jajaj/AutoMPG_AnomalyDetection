import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 1. Cargar y preprocesar el conjunto de datos Auto MPG
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
column_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin']
data = pd.read_csv(url, names=column_names, na_values='?', comment='\t', sep=' ', skipinitialspace=True)

# Manejar valores faltantes
data = data.dropna()

# Separar características
X = data.drop('mpg', axis=1).values

# Escalar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir datos (simulamos entrenamiento para Autoencoder)
X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)

# Simulación de Autoencoder (solo para comparación)
input_dim = X_train.shape[1]
autoencoder = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(input_dim,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(input_dim, activation='linear')
])
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_train, X_train, epochs=1, batch_size=32, verbose=0)  # Entrenamiento rápido
reconstructions = autoencoder.predict(X_scaled)
mse = np.mean(np.power(X_scaled - reconstructions, 2), axis=1)
threshold_autoencoder = np.percentile(mse, 95)

# k-NN
k = 5
nbrs = NearestNeighbors(n_neighbors=k).fit(X_scaled)
distances, _ = nbrs.kneighbors(X_scaled)
kth_distances = distances[:, -1]
threshold_knn = np.percentile(kth_distances, 95)

# 5. Comparar resultados
anomalies_autoencoder_full = mse > threshold_autoencoder
anomalies_knn = kth_distances > threshold_knn
common_anomalies = np.logical_and(anomalies_autoencoder_full, anomalies_knn)

print(f"Anomalías comunes entre Autoencoder y k-NN: {np.sum(common_anomalies)}")