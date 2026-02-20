#Red Neuronal para prediccion de valores
import numpy as np
import tensorflow as tf

x = np.array([10, 20, 30, 40, 50, 60], dtype=float).reshape(-1, 1) / 60
y = np.array([32, 62, 92, 122, 152, 182], dtype=float).reshape(-1, 1) / 182

modelo = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),
    tf.keras.layers.Dense(units=1)
])

modelo.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.01), loss='mean_squared_error')

print("Comenzando entrenamiento")
historial = modelo.fit(x, y, epochs=5000, verbose=0)
print("Modelo entrenado")

valores_a_predecir = np.array([15, 45], dtype=float).reshape(-1, 1)

predicciones = modelo.predict(valores_a_predecir)

for valor, pred in zip(valores_a_predecir.flatten(), predicciones):
    print(f"Valor: {valor} | Predicción: {pred[0]:.2f}")