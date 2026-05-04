#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 19 13:24:53 2025
@author: ym-t
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
import uhd

fs=56e6
b=fs
fc=2.45e9
t=0.05
muestras=int(fs*t)
gain=100

# =============================
# Detectar e inicializar USRP
# =============================
usrp = uhd.usrp.MultiUSRP()

usrp.set_rx_rate(fs, 0)
usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(fc, 0), 0)
usrp.set_rx_gain(gain, 0)

st_args = uhd.usrp.StreamArgs("fc32", "sc16")
st_args.channels = [0]
metadata = uhd.types.RXMetadata()
streamer = usrp.get_rx_stream(st_args)
recv_buffer = np.zeros(muestras, dtype=np.complex64)

# =============================
# Iniciar streaming y recibir
# =============================
stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
stream_cmd.num_samps = muestras
stream_cmd.stream_now = True
streamer.issue_stream_cmd(stream_cmd)

num_rx = streamer.recv(recv_buffer, metadata)

# Visualización simple de una parte de la señal
plt.figure(figsize=(10, 3))
plt.plot(np.real(recv_buffer[:1024]))
plt.title('Señal original (parte real)')
plt.xlabel('Índice')
plt.ylabel('Amplitud')
plt.grid(True)
plt.tight_layout()
plt.show()

# Welch: espectro completo (bidireccional) para señal IQ
nperseg = 1024
noverlap = 512
f, Pxx = welch(recv_buffer, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap, return_onesided=False)

# Reacomodamos el espectro para centrarlo
f = np.fft.fftshift(f)
Pxx = np.fft.fftshift(Pxx)

# Centrado en fc
f_mhz = f / 1e6 + fc / 1e6

# Graficamos
plt.figure(figsize=(10, 4))
plt.plot(f_mhz, Pxx)
plt.title('Densidad Espectral de Potencia usando Welch (centrado en 100 MHz)')
plt.xlabel('Frecuencia [MHz]')
plt.ylabel('PSD [V²/Hz]')
plt.grid(True)
plt.tight_layout()
plt.show()