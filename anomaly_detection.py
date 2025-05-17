import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Cargar y preprocesar el conjunto de datos Auto MPG
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
column_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin']
data = pd.read_csv(url, names=column_names, na_values='?', comment='\t', sep=' ', skipinitialspace=True)

# Manejar valores faltantes
data = data.dropna()

# Separar características (excluimos 'mpg' para enfocarnos en las características del automóvil)
X = data.drop('mpg', axis=1).values

# Escalar los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir en conjunto de entrenamiento (datos "normales")
X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)

# 2. Construir y entrenar el Autoencoder
input_dim = X_train.shape[1]  # Número de características (7)

# Definir el modelo Autoencoder
autoencoder = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(input_dim,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(input_dim, activation='linear')
])

autoencoder.compile(optimizer='adam', loss='mse')

# Entrenar el Autoencoder
history = autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)

# 3. Detectar anomalías con el Autoencoder
# Calcular el error de reconstrucción en el conjunto de prueba
reconstructions = autoencoder.predict(X_test)
mse = np.mean(np.power(X_test - reconstructions, 2), axis=1)

# Establecer un umbral para anomalías (percentil 95)
threshold_autoencoder = np.percentile(mse, 95)
anomalies_autoencoder = mse > threshold_autoencoder

print(f"Anomalías detectadas por Autoencoder: {np.sum(anomalies_autoencoder)}")