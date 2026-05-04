#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Espectro rápido con RTL‑SDR + SoapySDR
"""

import SoapySDR
from SoapySDR import *
import numpy as np
import matplotlib.pyplot as plt
import time

# ───────── Parámetros ─────────
fc   = 100.1e6          # Frecuencia central  (Hz)
fs   = 3.2e6            # Tasa de muestreo    (Hz)
gain = 40               # Ganancia (dB) – pon 'auto' si prefieres automática
nfft = 1024
num_muestras = 1024*1000  # 1 024 000 muestras (~0.43 s a 2.4 MS/s)

# ───────── Descubrir RTL‑SDR ─────────
disp = SoapySDR.Device.enumerate()
rtls = [d for d in disp if "driver=lime" in str(d)]
if not rtls:
    raise RuntimeError("❌ No hay RTL‑SDR conectado.")
    
sdr = SoapySDR.Device(rtls[0])
# ───────── Configura tu dongle ─────────
sdr.setSampleRate(SOAPY_SDR_RX, 0, fs)
sdr.setFrequency( SOAPY_SDR_RX, 0, fc)
sdr.setGain(      SOAPY_SDR_RX, 0, gain)   # quita si prefieres 'auto'
# ───────── Prepara el stream ─────────
rx = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
sdr.activateStream(rx)
# ───────── Captura ─────────
samples1 = np.empty(num_muestras, np.complex64)
sdr.readStream(rx, [samples1], num_muestras )
# ───────── Cierra todo ─────────
sdr.deactivateStream(rx)
sdr.closeStream(rx)
del sdr
# ───────── Espectro ─────────
plt.figure(figsize=(10,5))
plt.psd(samples1, NFFT=nfft, Fs=fs/1e6, Fc=fc/1e6)  # Fs y Fc en MHz
plt.xlabel("Frecuencia [MHz]")
plt.ylabel("PSD [dB/Hz]")
plt.title("Espectro recibido (RTL‑SDR)")
plt.tight_layout()
plt.show()
