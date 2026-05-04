import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

# Parámetros de PSD
Fs = 30e6   # Frecuencia de muestreo en Hz
NFFT = 1024  # Tamaño de la FFT

# Lista de archivos CSV y coordenadas
lista_tramas = [
    'muestras_1.csv',
    'muestras_2.csv',
    'muestras_3.csv',
    'muestras_4.csv',
    'muestras_5.csv',
    'muestras_6.csv'
]

coordenadas = [
    [-1, 0, 0],
    [-1, 5, 0],
    [0, -3, 0],
    [-8, -3, 0],
    [8, -3, 0],
    [-4, 5, 0]
]

def calcula_psd_valor_puntual(trama_csv, fs):
    df = pd.read_csv(trama_csv)
    # Extraer la señal compleja
    senal = df['real'].values
    senal_imag = df['imag'].values
    senal = senal + 1j * senal_imag

    plt.figure(figsize=(8, 6))
    pxx, freqs = plt.psd(senal, NFFT=1024, Fs=fs)
    # plt.show()
    plt.close()

    valor_puntual_db = 10*np.log10(pxx[512])
    print(f"Valor puntual PSD para {trama_csv}: {valor_puntual_db} dB")

    return valor_puntual_db


# Convertir coordenadas a array
coordenadas = np.array(coordenadas)

# Procesar cada archivo
listadevaloresmedios = []
for i in range(len(lista_tramas)):
    valor_psd_puntual = calcula_psd_valor_puntual(lista_tramas[i], Fs)
    listadevaloresmedios.append(valor_psd_puntual)
    coordenadas[i][2] = valor_psd_puntual  # Guardar valor puntual en coordenada Z

# Graficar
plt.figure(figsize=(8, 6))
sc = plt.scatter(
    coordenadas[:, 0],
    coordenadas[:, 1],
    c=coordenadas[:, 2],
    cmap='viridis',
    s=100,
    edgecolors='k'
)

plt.colorbar(sc, label='Valor puntual PSD (dB)')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Mapa de puntos (color según valor puntual en PSD)')
plt.grid(True)
plt.scatter(0, 0, color='blue', s=120, edgecolors='k', label='Origen', zorder=5)
plt.legend()
plt.show()

# Imprimir resultados
print("Coordenadas (X, Y, valor puntual PSD en dB):")
print(coordenadas)
print("Lista de valores puntuales (PSD en dB en índice medio):")
print(listadevaloresmedios)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interpolación espacial con IDW y Kriging usando coordenadas de tramas y PSD

@author: ym-t
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interpolación espacial con IDW y Kriging usando coordenadas de tramas y PSD
Origen de transmisión en (0,0)

@author: ym-t
"""

import numpy as np
import matplotlib.pyplot as plt
from pykrige.ok import OrdinaryKriging

# Coordenadas (X, Y) y valores Z (PSD de cada trama en dB)
# Reemplaza con tus valores reales

X = coordenadas[:, 0]
Y = coordenadas[:, 1]
Z = coordenadas[:, 2]

# Crear malla de interpolación
margin = 2  # margen extra alrededor de puntos
grid_x = np.linspace(np.min(X) - margin, np.max(X) + margin, 200)
grid_y = np.linspace(np.min(Y) - margin, np.max(Y) + margin, 200)
grid_x, grid_y = np.meshgrid(grid_x, grid_y)

# IDW
def idw(x, y, z, xi, yi, power=2):
    interpolated = np.zeros_like(xi)
    for i in range(xi.shape[0]):
        for j in range(xi.shape[1]):
            d = np.sqrt((x - xi[i, j])**2 + (y - yi[i, j])**2)
            if np.any(d == 0):
                interpolated[i, j] = z[d == 0][0]
            else:
                w = 1 / d**power
                interpolated[i, j] = np.sum(w * z) / np.sum(w)
    return interpolated

idw_result = idw(X, Y, Z, grid_x, grid_y)

# Kriging
try:
    OK = OrdinaryKriging(
        X, Y, Z,
        variogram_model='spherical',
        variogram_parameters={'sill': np.var(Z), 'range': 5, 'nugget': 0.1},
        verbose=False,
        enable_plotting=False,
        coordinates_type='euclidean'
    )
    kriging_result, ss = OK.execute('grid', grid_x[0, :], grid_y[:, 0])
    if np.any(np.isnan(kriging_result)):
        mean_val = np.nanmean(kriging_result)
        kriging_result = np.nan_to_num(kriging_result, nan=mean_val)
except Exception as e:
    print(f"Error en Kriging: {e}")
    kriging_result = np.full(grid_x.shape, np.mean(Z))

# Gráficos
fig, axs = plt.subplots(1, 3, figsize=(18, 6))

# Puntos originales
sc0 = axs[0].scatter(X, Y, c=Z, cmap='jet', s=80, edgecolors='black')
axs[0].scatter(0, 0, color='blue', marker='*', s=150, label='Origen transmisión', zorder=5)
axs[0].set_title("Puntos originales (PSD en dB)")
axs[0].set_xlabel("X")
axs[0].set_ylabel("Y")
axs[0].legend()
plt.colorbar(sc0, ax=axs[0])

# IDW
c1 = axs[1].imshow(idw_result,
                   extent=(grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()),
                   origin='lower',
                   cmap='jet')
axs[1].scatter(X, Y, c='white', edgecolors='black', s=30)
axs[1].scatter(0, 0, color='blue', marker='*', s=150, label='Origen transmisión', zorder=5)
axs[1].set_title("Interpolación IDW (PSD en dB)")
axs[1].legend()
plt.colorbar(c1, ax=axs[1])

# Kriging
c2 = axs[2].imshow(kriging_result,
                   extent=(grid_x.min(), grid_x.max(), grid_y.min(), grid_y.max()),
                   origin='lower',
                   cmap='jet')
axs[2].scatter(X, Y, c='white', edgecolors='black', s=30)
axs[2].scatter(0, 0, color='blue', marker='*', s=150, label='Origen transmisión', zorder=5)
axs[2].set_title("Interpolación Kriging (PSD en dB)")
axs[2].legend()
plt.colorbar(c2, ax=axs[2])

plt.tight_layout()
plt.show()

