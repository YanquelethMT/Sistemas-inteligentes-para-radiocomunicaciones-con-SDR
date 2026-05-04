#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 19 16:00:42 2025

@author: ym-t
"""

import numpy as np
import matplotlib.pyplot as plt
from rtlsdr import RtlSdr
from scipy.signal import decimate, welch

# Inicializa SDR para LTE (~850 MHz por ejemplo)
sdr = RtlSdr()
sdr.sample_rate = 3.2e6
sdr.center_freq = 915e6
sdr.gain = 'auto'

# Parámetros
block_time = 0.05  # 50 ms
num_samples = 1024*10



# Modo interactivo
plt.ion()

fig1, ax1 = plt.subplots(figsize=(10, 4))
line1, = ax1.plot([], [])
ax1.set_title('Densidad Espectral de Potencia (dB) - Sin Filtrar')
ax1.set_xlabel('Frecuencia [MHz]')
ax1.set_ylabel('PSD [dB]')
ax1.grid(True)

fig2, ax2 = plt.subplots(figsize=(6, 6))
scatter = ax2.scatter([], [], s=1, alpha=0.6)
ax2.set_title('Plano IQ - Sin Filtrar')
ax2.set_xlabel('Parte Real (I)')
ax2.set_ylabel('Parte Imaginaria (Q)')
ax2.grid(True)
ax2.set_aspect('equal', adjustable='datalim')

try:
    while True:
        samples = sdr.read_samples(num_samples)

       
        # PSD
        f, Pxx = welch(samples, fs=sdr.sample_rate, window='hann', nperseg=1024, noverlap=512, return_onesided=False)
        Pxx_dB = 10 * np.log10(Pxx + 1e-12)
        f=f+sdr.center_freq 
        line1.set_data(f / 1e6, Pxx_dB)
        ax1.relim()
        ax1.autoscale_view()

        # Plano IQ
        scatter.set_offsets(np.c_[np.real(samples), np.imag(samples)])
        ax2.relim()
        ax2.autoscale_view()

        fig1.canvas.draw()
        fig1.canvas.flush_events()
        fig2.canvas.draw()
        fig2.canvas.flush_events()

        plt.pause(0.01)

except KeyboardInterrupt:
    print("Interrumpido por el usuario")

finally:
    sdr.close()
    plt.ioff()
    plt.show()
