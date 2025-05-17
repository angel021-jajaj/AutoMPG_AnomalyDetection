import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

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

# Dividir en conjunto de entrenamiento
X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)

# 2. Construir y entrenar el Autoencoder
input_dim = X_train.shape[1]
autoencoder = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(input_dim,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(input_dim, activation='linear')
])
autoencoder.compile(optimizer='adam', loss='mse')
history = autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# 3. Detectar anomalías con el Autoencoder
reconstructions = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - reconstructions, 2), axis=1)
threshold_autoencoder = np.percentile(mse, 95)
anomalies_autoencoder = mse > threshold_autoencoder
print(f"Anomalías detectadas por Autoencoder: {np.sum(anomalies_autoencoder)}")

# 4. Detectar anomalías con k-NN
k = 5
nbrs = NearestNeighbors(n_neighbors=k).fit(X_scaled)
distances, _ = nbrs.kneighbors(X_scaled)
kth_distances = distances[:, -1]
threshold_knn = np.percentile(kth_distances, 95)
anomalies_knn = kth_distances > threshold_knn
print(f"Anomalías detectadas por k-NN: {np.sum(anomalies_knn)}")

# 5. Comparar resultados
anomalies_autoencoder_full = np.mean(np.power(X_scaled - autoencoder.predict(X_scaled), 2), axis=1) > threshold_autoencoder
common_anomalies = np.logical_and(anomalies_autoencoder_full, anomalies_knn)
print(f"Anomalías comunes entre Autoencoder y k-NN: {np.sum(common_anomalies)}")

# 6. Visualización
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.hist(mse, bins=50, alpha=0.5, label='Error de Reconstrucción')
plt.axvline(threshold_autoencoder, color='r', linestyle='--', label='Umbral')
plt.title('Autoencoder: Error de Reconstrucción')
plt.xlabel('MSE')
plt.ylabel('Frecuencia')
plt.legend()
plt.subplot(1, 2, 2)
plt.hist(kth_distances, bins=50, alpha=0.5, label='Distancia al k-ésimo vecino')
plt.axvline(threshold_knn, color='r', linestyle='--', label='Umbral')
plt.title('k-NN: Distancias')
plt.xlabel('Distancia')
plt.ylabel('Frecuencia')
plt.legend()
plt.tight_layout()
plt.show()