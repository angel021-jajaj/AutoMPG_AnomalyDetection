import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

# Cargar datos
column_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin']
data = pd.read_csv("auto-mpg.data", names=column_names, na_values='?', comment='\t', sep=' ', skipinitialspace=True)
data = data.dropna()
X = data.drop('mpg', axis=1).values

# Escalar datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)

# Autoencoder (simplificado)
input_dim = X_train.shape[1]
autoencoder = tf.keras.models.Sequential([
    tf.keras.layers.Dense(8, activation='relu', input_shape=(input_dim,)),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(input_dim, activation='linear')
])
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_train, X_train, epochs=5, batch_size=32, validation_split=0.2, verbose=1)

# Detectar anomalías con Autoencoder
reconstructions = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - reconstructions, 2), axis=1)
threshold_auto = np.percentile(mse, 95)
anomalies_auto = mse > threshold_auto
print(f"Anomalías por Autoencoder: {np.sum(anomalies_auto)}")

# k-NN
k = 5
nbrs = NearestNeighbors(n_neighbors=k).fit(X_scaled)
distances, _ = nbrs.kneighbors(X_scaled)
kth_distances = distances[:, -1]
threshold_knn = np.percentile(kth_distances, 95)
anomalies_knn = kth_distances > threshold_knn
print(f"Anomalías por k-NN: {np.sum(anomalies_knn)}")

# Comparar
common_anomalies = np.logical_and(anomalies_auto, anomalies_knn)
print(f"Anomalías comunes: {np.sum(common_anomalies)}")

# Gráficos
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(mse, bins=20, color='blue')
plt.axvline(threshold_auto, color='red', linestyle='--')
plt.title('Autoencoder - Error')
plt.subplot(1, 2, 2)
plt.hist(kth_distances, bins=20, color='green')
plt.axvline(threshold_knn, color='red', linestyle='--')
plt.title('k-NN - Distancias')
plt.show()