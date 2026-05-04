#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 15:24:41 2026

@author: ymt
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift, fftfreq

# =========================
# Parámetros de la señal
# =========================
B = 50        # ancho de banda [Hz]
f0 = 40       # frecuencia principal dentro del ancho de banda
t_max = 0.25     # duración [s]

# Señal continua (simulada con alta fs)
fs_cont = 1000
t = np.arange(0, t_max, 1/fs_cont)
x = np.sin(2*np.pi*f0*t)

# =========================
# Función para muestrear y graficar
# =========================
def analizar_muestreo(fs, titulo):
    
    # Muestreo
    Ts = 1/fs
    t_s = np.arange(0, t_max, Ts)
    x_s = np.sin(2*np.pi*f0*t_s)
    
    # FFT
    N = len(x_s)
    X = fftshift(fft(x_s))
    f = fftshift(fftfreq(N, d=Ts))
    
    # Gráfica
    plt.figure(figsize=(10,4))
    
    plt.subplot(1,2,1)
    plt.plot(t, x, label='Señal continua', alpha=0.5)
    plt.plot(t_s, x_s, 'r-o', markersize=5, label='Muestras')
    plt.title(f"Muestreo: fs = {fs} Hz\n{titulo}")
    plt.xlabel("Tiempo [s]")
    plt.legend()
    
    plt.subplot(1,2,2)

    # Espectro
    plt.plot(f, np.abs(X))
    
    # Encontrar el pico máximo
    idx_max = np.argmax(np.abs(X))
    f_max = f[idx_max]
    
    # Línea vertical punteada en el pico
    plt.axvline(x=f_max, linestyle='--', linewidth=1)
    
    # Opcional: línea horizontal en la amplitud máxima
    plt.axhline(y=np.abs(X[idx_max]), linestyle='--', linewidth=1)
    
    # Grid
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Labels
    plt.title("Espectro")
    plt.xlabel("Frecuencia [Hz]")
    plt.xlim(-100, 100)
    plt.tight_layout()
    plt.show()

# =========================
# Casos
# =========================

# 1. fs > 2B (correcto)
analizar_muestreo(fs=200, titulo="Sin aliasing (fs > 2B)")

# 2. fs = 2B (límite)
analizar_muestreo(fs=100, titulo="Caso crítico (fs = 2B)")

# 3. fs < 2B (aliasing)
analizar_muestreo(fs=60, titulo="Alias (fs < 2B)")