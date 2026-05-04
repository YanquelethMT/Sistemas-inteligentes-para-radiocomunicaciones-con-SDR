#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 12:09:22 2026

@author: yanqueleth
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp, spectrogram, welch

# ============================================
# Parámetros de la señal
# ============================================
fs = 2000               # Frecuencia de muestreo [Hz]
T = 2                   # Duración total [s]
t = np.linspace(0, T, int(fs * T), endpoint=False)

# Up-chirp lineal
f0 = 1                  # Frecuencia inicial [Hz]
f1 = 1000               # Frecuencia final [Hz]
x = chirp(t, f0=f0, f1=f1, t1=T, method='linear')

# ============================================
# Número de segmentos deseado
# ============================================
num_segmentos = 10       # Cambia este número libremente

# ============================================
# Dividir la señal en num_segmentos
# ============================================
N = len(x)
L = N // num_segmentos  # Longitud de cada segmento en muestras

segmentos = []
tiempos_seg = []

for i in range(num_segmentos):
    ini = i * L
    fin = (i + 1) * L if i < num_segmentos - 1 else N
    segmentos.append(x[ini:fin])
    tiempos_seg.append(t[ini:fin])

# ============================================
# Crear figura dinámica:
# 1 fila para señal completa + espectrograma
# num_segmentos filas para segmentos + PSD
# ============================================
num_filas = num_segmentos + 1
fig, axes = plt.subplots(num_filas, 2, figsize=(12, 3.2 * num_filas))
plt.subplots_adjust(hspace=0.8, wspace=0.35)

# Si solo hay una fila extraña, asegurar índice 2D
if num_filas == 1:
    axes = np.array([axes])

# ============================================
# Imagen 1: señal completa
# ============================================
axes[0, 0].plot(t, x)
axes[0, 0].set_title("1. Up-chirp completo")
axes[0, 0].set_xlabel("Tiempo [s]")
axes[0, 0].set_ylabel("Amplitud")
axes[0, 0].grid(True)

# ============================================
# Imagen 2: espectrograma completo
# ============================================
f_spec, t_spec, Sxx = spectrogram(
    x, fs=fs, nperseg=256, noverlap=200, scaling='density', mode='psd'
)
im = axes[0, 1].pcolormesh(
    t_spec, f_spec, 10 * np.log10(Sxx + 1e-12), shading='gouraud'
)
axes[0, 1].set_title("2. Espectrograma del up-chirp")
axes[0, 1].set_xlabel("Tiempo [s]")
axes[0, 1].set_ylabel("Frecuencia [Hz]")
cbar = fig.colorbar(im, ax=axes[0, 1])
cbar.set_label("PSD [dB/Hz]")

# ============================================
# Imágenes restantes: segmentos y PSD
# ============================================
for i in range(num_segmentos):
    fila = i + 1

    # Señal del segmento
    axes[fila, 0].plot(tiempos_seg[i], segmentos[i])
    axes[fila, 0].set_title(f"Segmento {i+1} del chirp")
    axes[fila, 0].set_xlabel("Tiempo [s]")
    axes[fila, 0].set_ylabel("Amplitud")
    axes[fila, 0].grid(True)

    # PSD del segmento con Welch
    nperseg_welch = min(256, len(segmentos[i]))
    f_welch, Pxx = welch(segmentos[i], fs=fs, nperseg=nperseg_welch)

    axes[fila, 1].plot(f_welch, 10 * np.log10(Pxx + 1e-12))
    axes[fila, 1].set_title(f"PSD del segmento {i+1}")
    axes[fila, 1].set_xlabel("Frecuencia [Hz]")
    axes[fila, 1].set_ylabel("PSD [dB/Hz]")
    axes[fila, 1].grid(True)

plt.tight_layout()
plt.show()


# Segmentos y PSD
frecuencias_psd = None
matriz_psd = []

for i in range(num_segmentos):
    fila = i + 1

    # Señal del segmento
    axes[fila, 0].plot(tiempos_seg[i], segmentos[i])
    axes[fila, 0].set_title(f"Segmento {i+1} del chirp")
    axes[fila, 0].set_xlabel("Tiempo [s]")
    axes[fila, 0].set_ylabel("Amplitud")
    axes[fila, 0].grid(True)

    # PSD del segmento
    nperseg_welch = min(256, len(segmentos[i]))
    f_welch, Pxx = welch(segmentos[i], fs=fs, nperseg=nperseg_welch)
    Pxx_dB = 10 * np.log10(Pxx + 1e-12)

    axes[fila, 1].plot(f_welch, Pxx_dB)
    axes[fila, 1].set_title(f"PSD del segmento {i+1}")
    axes[fila, 1].set_xlabel("Frecuencia [Hz]")
    axes[fila, 1].set_ylabel("PSD [dB/Hz]")
    axes[fila, 1].grid(True)

    if frecuencias_psd is None:
        frecuencias_psd = f_welch
    matriz_psd.append(Pxx_dB)

plt.tight_layout()
plt.show()

# ============================================
# Representación 3D de las PSD por segmento
# ============================================
matriz_psd = np.array(matriz_psd)   # dimensiones: [num_segmentos, num_frec]
segmentos_idx = np.arange(1, num_segmentos + 1)




# ============================================
# Representación 3D de las PSD por segmento
# + línea de frecuencia dominante
# ============================================
matriz_psd = np.array(matriz_psd)   # [num_segmentos, num_frec]

# Posiciones de segmentos para arrancar en 0
segmentos_idx = np.arange(num_segmentos)

# Malla para superficie
F, S = np.meshgrid(frecuencias_psd, segmentos_idx)

fig3d = plt.figure(figsize=(12, 7))
ax3d = fig3d.add_subplot(111, projection='3d')

# Superficie
surf = ax3d.plot_surface(
    F, S, matriz_psd,
    cmap='viridis',
    edgecolor='none',
    alpha=0.9
)

# ============================================
# Línea de frecuencia dominante por segmento
# ============================================
freq_dom = []
psd_dom = []

for i in range(num_segmentos):
    idx_max = np.argmax(matriz_psd[i, :])
    freq_dom.append(frecuencias_psd[idx_max])
    psd_dom.append(matriz_psd[i, idx_max])

freq_dom = np.array(freq_dom)
psd_dom = np.array(psd_dom)

# Línea 3D encima de la superficie
ax3d.plot(
    freq_dom,               # X = frecuencia dominante
    segmentos_idx,          # Y = segmento
    psd_dom + 1.0,          # Z = un poco arriba de la superficie
    color='red',
    linewidth=3,
    marker='o',
    markersize=5,
    label='Frecuencia dominante'
)

# ============================================
# Ejes y formato
# ============================================
ax3d.set_title("PSD 3D por segmento + frecuencia dominante")
ax3d.set_xlabel("Frecuencia [Hz]")
ax3d.set_ylabel("Segmento")
ax3d.set_zlabel("PSD [dB/Hz]")

# Forzar origen visual común
ax3d.set_xlim3d(0, frecuencias_psd.max())
ax3d.set_ylim3d(0, num_segmentos - 1)

# Mostrar segmentos como 1,2,3,...
ax3d.set_yticks(np.arange(num_segmentos))
ax3d.set_yticklabels(np.arange(1, num_segmentos + 1))

fig3d.colorbar(surf, ax=ax3d, shrink=0.7, pad=0.1, label="PSD [dB/Hz]")

ax3d.legend()
plt.tight_layout()
plt.show()