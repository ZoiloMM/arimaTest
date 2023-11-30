import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Cargar los datos del archivo CSV, que contiene la mitad de los datos
data = pd.read_csv("fs140.csv")
data = data.loc[data['MontoGasto'] >= 7000]
print(data['MontoGasto'].size)
print(data['MontoGasto'].min())
# Cargar el segundo DataFrame, que contiene toda los datos
data2 = pd.read_csv("datos.csv")
data2 = data2.loc[data2['MontoGasto'] >= 7000]
print(data2['MontoGasto'].size)
# Convertir la columna de fechas en un índice de tiempo
data['FechaEfectiva'] = pd.to_datetime(data['FechaEfectiva']).dt.strftime('%Y-%m-%d')
data.set_index('FechaEfectiva', inplace=True)

# Convertir la columna de fechas del segundo DataFrame en un índice de tiempo
data2['FechaEfectiva'] = pd.to_datetime(data2['FechaEfectiva']).dt.strftime('%Y-%m-%d')
data2.set_index('FechaEfectiva', inplace=True)

# Ajustar el modelo ARIMA (ajusta los parámetros p, d, q según corresponda) para data
model = ARIMA(data['MontoGasto'], order=(21, 0, 16))
results = model.fit()

# Número de pasos a predecir
nforecast = 20  # Cambia esto al número de pasos que deseas predecir

# Realizar las predicciones para data
predict = results.get_prediction(end=len(data) + nforecast - 1)
predict_mean = predict.predicted_mean
predict_ci = predict.conf_int(alpha=0.5)

print(predict_mean)
#print(predict_mean)

#print(predict_ci)

'''
# Crear un índice numérico para las predicciones
index = np.arange(len(data) + nforecast)

# Configurar el tamaño de la figura (hazla más ancha)
fig, ax = plt.subplots(figsize=(15, 6))

# Graficar los datos originales y las predicciones para data
ax.plot(index[:-nforecast], data['MontoGasto'], label="Original de 0 al 140", color="blue")
ax.plot(index[-nforecast:], predict_mean[-nforecast:], 'red', label="Predicciones (Data)", linestyle='--')
ax.fill_between(index, predict_ci.iloc[:, 0], predict_ci.iloc[:, 1], alpha=0.15, label="Intervalo de confianza (Data)")

# Graficar los datos originales para data2
ax.plot(data2.index, data2['MontoGasto'], label="Original de 0 al 160", color="violet")

# Configurar el título y las etiquetas de los ejes
ax.set_title("Metodo ARIMA, Coeficientes p: 21, d: 0 y q: 16")
ax.set_xlabel("Período / Fecha")
ax.set_ylabel("MontoGasto")
ax.legend()

# Mostrar el gráfico
plt.tight_layout()  # Ajusta automáticamente el espacio entre las gráficas
plt.show()
'''