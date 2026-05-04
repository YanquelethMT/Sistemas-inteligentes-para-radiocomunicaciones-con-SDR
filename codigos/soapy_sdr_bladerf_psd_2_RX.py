#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lectura continua de los 2 canales RX de un BladeRF usando SoapySDR.
Calcula y muestra la PSD de ambos canales en cada ciclo del while.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
import SoapySDR
from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32
import sys

# ==========================================
# CONFIGURACIÓN
# ==========================================
fs = 40e6
duration = 0.05
num_samples = int(fs * duration)
fc_inicial = 2.4e9
center_freq = fc_inicial + fs/2
gain = 10
NFFT = 1024

# ==========================================
# Abrir dispositivo BladeRF
# ==========================================
try:
    dev = SoapySDR.Device("driver=bladerf")
except Exception as e:
    print("Error abriendo BladeRF:", e)
    sys.exit(1)

print("✅ Dispositivo abierto:", dev.getDriverKey(), "-", dev.getHardwareKey())

# Configurar canales
channels = [0, 1]
for ch in channels:
    dev.setFrequency(SOAPY_SDR_RX, ch, center_freq)
    dev.setGain(SOAPY_SDR_RX, ch, gain)
    dev.setSampleRate(SOAPY_SDR_RX, ch, fs)
    dev.setBandwidth(SOAPY_SDR_RX, ch, fs)

# Preparar stream una sola vez
stream = dev.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, channels)
dev.activateStream(stream)
print("🎧 Capturando en tiempo real. Ctrl+C para detener.\n")

plt.ion()
plt.figure(figsize=(10,6))

try:
    while True:
        # Buffers
        bufs = [np.zeros(num_samples, dtype=np.complex64) for _ in channels]

        # Leer muestras
        sr = dev.readStream(stream, bufs, num_samples)
        ret = sr.ret
        if ret <= 0:
            print("⚠️ Error o timeout en readStream:", ret)
            continue

        # Calcular PSD por canal
        plt.clf()
        for idx, buf in enumerate(bufs):
            buf = buf[:ret]
            f, Pxx = welch(buf, fs=fs, nperseg=NFFT, return_onesided=False, scaling='density')
            f = np.fft.fftshift(f) + center_freq
            Pxx = np.fft.fftshift(Pxx)
            Pxx = 10 * np.log10(Pxx/1e-3 + 1e-12)

            plt.subplot(len(bufs), 1, idx+1)
            plt.plot(f/1e6, Pxx)
            plt.title(f"Canal {idx} - PSD (Welch)")
            plt.xlabel("Frecuencia [MHz]")
            plt.ylabel("PSD [dB/Hz]")
            plt.grid(True)

        plt.tight_layout()
        plt.pause(0.001)

except KeyboardInterrupt:
    print("\n🛑 Captura detenida por el usuario.")

finally:
    dev.deactivateStream(stream)
    dev.closeStream(stream)
    plt.close()
    print("✅ Stream cerrado correctamente.")
