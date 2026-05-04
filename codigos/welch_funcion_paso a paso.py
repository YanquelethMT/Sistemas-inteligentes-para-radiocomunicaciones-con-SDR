#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 19 13:39:37 2025

@author: ym-t
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import hann
from rtlsdr import RtlSdr

# Configuración del SDR
sdr = RtlSdr()
sdr.sample_rate = 2.4e6
sdr.center_freq = 100e6
sdr.gain = 'auto'

# Lectura de muestras
samples = sdr.read_samples(1000*1024)
sdr.close()

# Parámetros
nperseg = 1024
noverlap = 512
step = nperseg - noverlap
window = hann(nperseg)
fs = sdr.sample_rate
fc = sdr.center_freq

# Número total de ventanas
num_segments = (len(samples) - noverlap) // step

# Almacenamos las PSDs
psd_list = []

# Iteramos sobre cada segmento
plt.figure(figsize=(10, 4))

for i in range(num_segments):
    start = i * step
    end = start + nperseg
    segment = samples[start:end]

    if len(segment) < nperseg:
        break  # evitamos segmentos incompletos

    segment = segment * window  # Aplicar ventana

    # FFT y PSD (Potencia por Hz)
    fft_vals = np.fft.fftshift(np.fft.fft(segment))
    psd = np.abs(fft_vals) ** 2 / (fs * np.sum(window**2))
    
    psd_list.append(psd)

    # Eje de frecuencia centrado
    f = np.fft.fftshift(np.fft.fftfreq(nperseg, 1/fs))
    f_mhz = f / 1e6 + fc / 1e6
    
    if i>num_segments-100:
        # Graficamos el espectro de esta ventana
        # plt.semilogy(f_mhz, psd, alpha=0.3)
        print(i)
        plt.plot(f_mhz, 10*np.log10(psd), alpha=0.3, color='gray')

        


# Promedio de todas las PSDs (equivalente a Welch)
Pxx_avg = np.mean(psd_list, axis=0)

# Graficamos la PSD promedio
plt.plot(f_mhz, 10*np.log10(Pxx_avg), color='blue', linewidth=2, label='PSD promedio')
plt.title('PSD promedio (método de Welch manual)')
plt.xlabel('Frecuencia [MHz]')
plt.ylabel('PSD [dB]')
plt.grid(True)
plt.tight_layout()
plt.show()
