#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 15:21:49 2026

@author: ymt
"""

import numpy as np
import uhd
import time
import matplotlib.pyplot as plt

# =========================
# Parámetros del chirp
# =========================
fc = 5.81e9          # Frecuencia central RF [Hz]
bw = 40e6            # Ancho de banda del chirp [Hz]
fs = 56e6            # Tasa de muestreo [Hz]
tx_gain = 20         # Ajusta con cuidado
chirp_time = 10e-3    # Duración de cada chirp [s]
num_chirps = 2000     # Número de chirps a transmitir
amplitude = 0.5      # Mantener < 1 para evitar clipping

# Barrido en banda base: de -bw/2 a +bw/2
f0 = -bw / 2
f1 =  bw / 2

# =========================
# Crear chirp complejo
# =========================
N = int(fs * chirp_time)
t = np.arange(N) / fs

# Chirp lineal:
# f(t) = f0 + k*t, con k = (f1 - f0)/T
k = (f1 - f0) / chirp_time

# Fase instantánea:
# phi(t) = 2*pi*(f0*t + 0.5*k*t^2)
phi = 2 * np.pi * (f0 * t + 0.5 * k * t**2)

# Señal compleja IQ
chirp_bb = amplitude * np.exp(1j * phi).astype(np.complex64)

# Repetir varios chirps en un solo buffer
tx_signal = np.tile(chirp_bb, num_chirps).astype(np.complex64)

# =========================
# Configurar USRP
# =========================
usrp = uhd.usrp.MultiUSRP()

usrp.set_tx_rate(fs)
usrp.set_tx_freq(uhd.libpyuhd.types.tune_request(fc), 0)
usrp.set_tx_gain(tx_gain, 0)
usrp.set_tx_bandwidth(bw, 0)

# Esperar a que el LO se asiente
time.sleep(0.2)

print("TX rate       :", usrp.get_tx_rate())
print("TX freq       :", usrp.get_tx_freq())
print("TX gain       :", usrp.get_tx_gain())
print("TX bandwidth  :", usrp.get_tx_bandwidth())

# =========================
# Crear streamer TX
# =========================
st_args = uhd.usrp.StreamArgs("fc32", "sc16")
st_args.channels = [0]
tx_streamer = usrp.get_tx_stream(st_args)

md = uhd.types.TXMetadata()
md.start_of_burst = True
md.end_of_burst = False
md.has_time_spec = False

# =========================
# Transmitir
# =========================
max_samps = tx_streamer.get_max_num_samps()
idx = 0
total = len(tx_signal)

while idx < total:
    chunk = tx_signal[idx: idx + max_samps]
    tx_streamer.send(chunk, md)
    md.start_of_burst = False
    idx += len(chunk)

# Marcar fin de ráfaga
md.end_of_burst = True
tx_streamer.send(np.zeros((1,), dtype=np.complex64), md)

print("Transmisión terminada.")

plt.plot(chirp_bb.real)
plt.show()