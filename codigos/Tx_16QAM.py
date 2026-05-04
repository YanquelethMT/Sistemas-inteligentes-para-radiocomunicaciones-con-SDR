#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Transmisión de ráfagas 16-QAM con LimeSDR usando SoapySDR
@author: ym-t
"""

import SoapySDR
from SoapySDR import *
import numpy as np
import time
import matplotlib.pyplot as plt

# === Parámetros SDR ===
sample_rate = 2e6
center_freq = 915e6
bandwidth = 5e6
gain = 40
symbol_rate = 100e3
samples_per_symbol = int(sample_rate / symbol_rate)

burst_duration = 0.5  # 100 ms
total_duration = 100.0  # 3 segundos
num_bursts = int(total_duration / burst_duration)

# === Función para generar 16-QAM ===
def generate_16qam(num_symbols):
    symbols = np.random.randint(0, 16, num_symbols)
    mapping_table = {
        0: (-3 - 3j), 1: (-3 - 1j), 2: (-3 + 3j), 3: (-3 + 1j),
        4: (-1 - 3j), 5: (-1 - 1j), 6: (-1 + 3j), 7: (-1 + 1j),
        8: ( 3 - 3j), 9: ( 3 - 1j),10: ( 3 + 3j),11: ( 3 + 1j),
        12:( 1 - 3j),13: ( 1 - 1j),14: ( 1 + 3j),15: ( 1 + 1j)
    }
    modulated = np.array([mapping_table[sym] for sym in symbols])
    return modulated / np.sqrt(10)

# === Interpolación (upsample) ===
def upsample(signal, sps):
    return np.repeat(signal, sps)

# === Inicializar SDR ===
devices = SoapySDR.Device.enumerate()
if not devices:
    raise RuntimeError("No se detectó ningún dispositivo SDR")
args = devices[0]
sdr = SoapySDR.Device(args)
sdr.setSampleRate(SOAPY_SDR_TX, 0, sample_rate)
sdr.setFrequency(SOAPY_SDR_TX, 0, center_freq)
sdr.setBandwidth(SOAPY_SDR_TX, 0, bandwidth)
sdr.setGain(SOAPY_SDR_TX, 0, gain)
tx_stream = sdr.setupStream(SOAPY_SDR_TX, SOAPY_SDR_CF32)
sdr.activateStream(tx_stream)

# === Transmisión ===
num_samples_per_burst = int(sample_rate * burst_duration)
num_symbols = num_samples_per_burst // samples_per_symbol

print("Transmitiendo ráfagas 16-QAM cada 100 ms durante 3 segundos...")
start_time = time.time()

for i in range(num_bursts):
    qam_symbols = generate_16qam(num_symbols)
    tx_signal = upsample(qam_symbols, samples_per_symbol).astype(np.complex64)

    sr = sdr.writeStream(tx_stream, [tx_signal], len(tx_signal))
    if sr.ret != len(tx_signal):
        print(f"Error transmitiendo ráfaga {i+1}/{num_bursts}")

    elapsed = time.time() - start_time
    next_burst_time = (i + 1) * burst_duration
    sleep_time = next_burst_time - elapsed
    if sleep_time > 0:
        time.sleep(sleep_time)

sdr.deactivateStream(tx_stream)
sdr.closeStream(tx_stream)
print("Transmisión finalizada.")

# === Visualización de constelación generada (opcional) ===
qam_symbols = generate_16qam(1000)
tx_signal = upsample(qam_symbols, samples_per_symbol)
plt.figure(figsize=(6,6))
plt.scatter(np.real(tx_signal), np.imag(tx_signal), s=2, alpha=0.5)
plt.title('Constelación 16-QAM generada')
plt.xlabel('Parte Real (I)')
plt.ylabel('Parte Imaginaria (Q)')
plt.grid(True)
plt.axis('equal')
plt.show()
