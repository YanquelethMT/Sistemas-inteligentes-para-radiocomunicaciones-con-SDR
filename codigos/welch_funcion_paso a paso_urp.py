#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 12:51:01 2026

@author: ymt
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 19 13:39:37 2025

@author: ym-t
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import hann
import uhd

# Configuración del SDR
fs = 2.4e6
fc = 100e6
gain = 50
num_samples = 1000 * 1024

usrp = uhd.usrp.MultiUSRP()
usrp.set_rx_rate(fs, 0)
usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(fc, 0), 0)
usrp.set_rx_gain(gain, 0)

# Lectura de muestras
st_args = uhd.usrp.StreamArgs("fc32", "sc16")
st_args.channels = [0]
metadata = uhd.types.RXMetadata()
streamer = usrp.get_rx_stream(st_args)

samples = np.zeros(num_samples, dtype=np.complex64)

stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.num_done)
stream_cmd.num_samps = num_samples
stream_cmd.stream_now = True
streamer.issue_stream_cmd(stream_cmd)

num_rx = streamer.recv(samples, metadata)

# Parámetros
nperseg = 1024
noverlap = 512
step = nperseg - noverlap
window = hann(nperseg)

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
    
    if i > num_segments - 100:
        print(i)
        plt.plot(f_mhz, 10*np.log10(psd + 1e-12), alpha=0.3, color='gray')

# Promedio de todas las PSDs (equivalente a Welch)
Pxx_avg = np.mean(psd_list, axis=0)

# Graficamos la PSD promedio
plt.plot(f_mhz, 10*np.log10(Pxx_avg + 1e-12), color='blue', linewidth=2, label='PSD promedio')
plt.title('PSD promedio (método de Welch manual)')
plt.xlabel('Frecuencia [MHz]')
plt.ylabel('PSD [dB]')
plt.grid(True)
plt.tight_layout()
plt.show()