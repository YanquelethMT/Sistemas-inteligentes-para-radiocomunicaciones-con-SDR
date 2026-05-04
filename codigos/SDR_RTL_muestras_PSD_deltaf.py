#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 16:18:58 2026

@author: ymt
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from rtlsdr import RtlSdr
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# =========================
# Parámetros de adquisición
# =========================
fc = 100e6          # Frecuencia central [Hz]
fs = 2.4e6          # Frecuencia de muestreo [Hz]
gain = 20           # Ganancia [dB]
num_samples = 262144  # Número de muestras

# =========================
# Parámetros de PSD (Welch)
# =========================
nperseg = 1024
noverlap = 512

# =========================
# Configuración del SDR
# =========================
sdr = RtlSdr()

try:
    sdr.sample_rate = fs
    sdr.center_freq = fc
    sdr.gain = gain

    print("Adquiriendo muestras...")
    samples = sdr.read_samples(num_samples)
    print("Listo.")

    # =========================
    # Estimación de PSD (Welch)
    # =========================
    f, Pxx = signal.welch(
        samples,
        fs=fs,
        window='hann',
        nperseg=nperseg,
        noverlap=noverlap,
        return_onesided=False,   # IMPORTANTE para señales IQ
        scaling='density'
    )

    # =========================
    # Centrado en frecuencia
    # =========================
    f = np.fft.fftshift(f)
    Pxx = np.fft.fftshift(Pxx)

    # Convertir a dB
    Pxx_dB = 10 * np.log10(Pxx + 1e-12)

    # =========================
    # Eje de frecuencia absoluto
    # =========================
    f_axis = f + fc

    # =========================
    # Graficar PSD
    # =========================
    plt.figure(figsize=(10, 5))
    plt.plot(f_axis/1e6, Pxx_dB)
    plt.xlabel("Frecuencia [MHz]")
    plt.ylabel("PSD [dB/Hz]")
    plt.title("Densidad Espectral de Potencia (PSD) - RTL-SDR")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

finally:
    sdr.close()