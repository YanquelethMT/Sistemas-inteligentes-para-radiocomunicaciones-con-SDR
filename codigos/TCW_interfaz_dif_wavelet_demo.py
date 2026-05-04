#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 23:42:26 2026

@author: ymt
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons
import pywt

# 1. Creación de la señal con singularidades
t = np.linspace(0, 1, 1000)
signal = np.zeros_like(t)
signal[200:400] = 1.0   # Escalón
signal[600] = 3.0       # Pico Dirac
signal += 0.05 * np.random.randn(len(t)) # Un poco de ruido

# 2. Configuración inicial
scales = np.arange(1, 128)
wavelet_actual = 'mexh'

# 3. Función para calcular CWT y Esqueleto (Máximos)
def obtener_cwt(wav):
    coef, _ = pywt.cwt(signal, scales, wav)
    return np.abs(coef)

# 4. Configuración de la interfaz gráfica
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
plt.subplots_adjust(left=0.1, bottom=0.25)

# Gráfico de la señal
line_sig, = ax1.plot(t, signal, color='gray', alpha=0.5, label='Señal')
line_wav, = ax1.plot(t, np.zeros_like(t), color='red', lw=2, label='Wavelet Madre')
ax1.legend(loc='upper right')
ax1.set_title("Análisis de Singularidades (Desplazamiento y Escala)")
# --- AÑADE ESTA LÍNEA AQUÍ ---
ax1.set_ylim(-1, 3.5) 
# -----------------------------
# Gráfico del Espectrograma (CWT)
coef_data = obtener_cwt(wavelet_actual)
img = ax2.imshow(coef_data, extent=[0, 1, 128, 1], cmap='jet', aspect='auto')
line_b = ax2.axvline(x=0.5, color='white', linestyle='--') # Línea de desplazamiento
line_a = ax2.axhline(y=30, color='white', linestyle='--')  # Línea de escala
fig.colorbar(img, ax=ax2, label='Módulo CWT')
ax2.set_ylabel("Escala (a)")
ax2.set_xlabel("Desplazamiento (b)")

# 5. Creación de Sliders
ax_b = plt.axes([0.15, 0.1, 0.65, 0.03])
ax_a = plt.axes([0.15, 0.05, 0.65, 0.03])
s_b = Slider(ax_b, 'Desplazamiento (b)', 0, 1.0, valinit=0.5)
s_a = Slider(ax_a, 'Escala (a)', 1, 120, valinit=30)

# 6. Función de actualización
def update(val):
    b = s_b.val
    a = s_a.val
    
    # Actualizar líneas guía en el mapa CWT
    line_b.set_xdata([b, b])
    line_a.set_ydata([a, a])
    
    # Generar la forma de la wavelet para mostrarla arriba
    # b_idx es la posición en el array
    wav_func, _ = pywt.ContinuousWavelet(wavelet_actual).wavefun(level=8)
    wav_t = np.linspace(-5, 5, len(wav_func)) * a / 100 # Escalamiento visual
    
    # Ajustar posición de la wavelet roja
    line_wav.set_xdata(wav_t + b)
    line_wav.set_ydata(wav_func * (1/np.sqrt(a)) * 2) # Normalización de energía
    
    fig.canvas.draw_idle()

s_b.on_changed(update)
s_a.on_changed(update)

# Botones para cambiar la Wavelet Madre
rax = plt.axes([0.85, 0.05, 0.12, 0.1])
radio = RadioButtons(rax, ('mexh', 'morl', 'cgau1'))

def change_wav(label):
    global wavelet_actual
    wavelet_actual = label
    img.set_data(obtener_cwt(label))
    update(None)

radio.on_clicked(change_wav)

update(None)
plt.show()