#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 15:58:21 2026

@author: ymt
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from rtlsdr import RtlSdr
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Parámetros de adquisición
# =========================
fc = 100e6          # Frecuencia central [Hz]
fs = 2.4e6          # Frecuencia de muestreo [Hz]
gain = 20           # Ganancia [dB] (puede variar según tu RTL)
num_samples = 262144  # Número de muestras complejas a capturar

# =========================
# Configuración del RTL-SDR
# =========================
sdr = RtlSdr()

try:
    sdr.sample_rate = fs
    sdr.center_freq = fc
    sdr.gain = gain

    print("Configuración del SDR:")
    print(f"  Frecuencia central: {sdr.center_freq/1e6:.3f} MHz")
    print(f"  Frecuencia de muestreo: {sdr.sample_rate/1e6:.3f} MS/s")
    print(f"  Ganancia: {sdr.gain} dB")
    print(f"  Número de muestras a capturar: {num_samples}")

    # =========================
    # Adquisición de muestras IQ
    # =========================
    samples = sdr.read_samples(num_samples)

    print("\nCaptura completada.")
    print(f"Tipo de dato: {samples.dtype}")
    print(f"Tamaño del arreglo: {samples.shape}")
    
    # =========================
    # Graficar parte real e imaginaria
    # =========================
   
finally:
    sdr.close()