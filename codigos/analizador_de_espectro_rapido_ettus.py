#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 10:19:22 2026

@author: ymt
"""

import numpy as np
import matplotlib.pyplot as plt
import uhd
from scipy.signal import welch
import time

# =========================================================
# Parámetros de usuario
# =========================================================
fc = 5.8e9      # Frecuencia central [Hz]
fs = 50e6             # Frecuencia de muestreo [Hz]
bw = fs               # Ancho de banda RX [Hz]
gain = 100             # Ganancia RX [dB]
fft_size = 2048       # Tamaño base para Welch
tiempo = 0.005
muestras = int(fs * tiempo)
device_args = "type=b200"
antenna = "TX/RX"
channel = 0

# Parámetros del espectrograma
historial = 100       # Número de líneas visibles en el espectrograma

# =========================================================
# Inicialización del USRP
# =========================================================
print("[INFO] Abriendo USRP...")
usrp = uhd.usrp.MultiUSRP(device_args)

usrp.set_rx_rate(fs, channel)
usrp.set_rx_freq(fc, channel)
usrp.set_rx_gain(gain, channel)

try:
    usrp.set_rx_bandwidth(bw, channel)
except Exception as e:
    print(f"[WARN] No se pudo configurar el ancho de banda: {e}")

try:
    usrp.set_rx_antenna(antenna, channel)
except Exception as e:
    print(f"[WARN] No se pudo configurar la antena: {e}")

print(f"[INFO] fs real: {usrp.get_rx_rate(channel)/1e6:.3f} MS/s")
print(f"[INFO] fc real: {usrp.get_rx_freq(channel)/1e6:.3f} MHz")
print(f"[INFO] bw real: {usrp.get_rx_bandwidth(channel)/1e6:.3f} MHz")
print(f"[INFO] gain real: {usrp.get_rx_gain(channel):.2f} dB")

# =========================================================
# Configuración del stream
# =========================================================
st_args = uhd.usrp.StreamArgs("fc32", "sc16")
st_args.channels = [channel]

rx_stream = usrp.get_rx_stream(st_args)
metadata = uhd.types.RXMetadata()

buffer = np.zeros((1, muestras), dtype=np.complex64)

stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
stream_cmd.stream_now = True
rx_stream.issue_stream_cmd(stream_cmd)

print("[INFO] Recibiendo...")

# =========================================================
# Parámetros de Welch
# =========================================================
nperseg = fft_size
noverlap = fft_size // 2
nfft = fft_size
return_onesided = False

# =========================================================
# Gráfica interactiva
# =========================================================
plt.ion()
plt.style.use('dark_background')

fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [1, 1.2]}
)

fig.patch.set_facecolor('black')
ax1.set_facecolor('black')
ax2.set_facecolor('black')

turquesa = '#00FFF7'
gris_grid = '#444444'
blanco = '#FFFFFF'

# -------------------------
# Subplot 1: PSD
# -------------------------
line_psd, = ax1.plot([], [], color=turquesa, linewidth=1.5)

ax1.set_title("PSD en tiempo real - Ettus B210 (Welch)", color=blanco)
ax1.set_xlabel("Frecuencia [MHz]", color=blanco)
ax1.set_ylabel("Potencia [dBm/Hz]", color=blanco)
ax1.set_ylim(-180, -50)
ax1.grid(True, color=gris_grid, linestyle='--', linewidth=0.5, alpha=0.7)
ax1.tick_params(axis='x', colors=blanco)
ax1.tick_params(axis='y', colors=blanco)

for spine in ax1.spines.values():
    spine.set_color(gris_grid)

# -------------------------
# Subplot 2: Espectrograma
# -------------------------
spectrogram_data = np.full((historial, nfft), -180.0)

freq_axis_temp = (np.fft.fftshift(np.fft.fftfreq(nfft, d=1/fs)) + fc) / 1e6

img = ax2.imshow(
    spectrogram_data,
    aspect='auto',
    cmap='hot',
    origin='lower',
    extent=[freq_axis_temp[0], freq_axis_temp[-1], 0, historial],
    vmin=-90,
    vmax=-40
)

ax2.set_title("Espectrograma en tiempo real", color=blanco)
ax2.set_xlabel("Frecuencia [MHz]", color=blanco)
ax2.set_ylabel("Tiempo", color=blanco)
ax2.tick_params(axis='x', colors=blanco)
ax2.tick_params(axis='y', colors=blanco)

for spine in ax2.spines.values():
    spine.set_color(gris_grid)

cbar = fig.colorbar(img, ax=ax2)
cbar.set_label("Potencia [dBm/Hz]", color=blanco)
cbar.ax.yaxis.set_tick_params(color=blanco)
plt.setp(cbar.ax.get_yticklabels(), color=blanco)

plt.tight_layout()

# =========================================================
# Título dinámico
# =========================================================
start_time = time.time()

titulo_global = fig.suptitle(
    "Sensing!!!",
    color=blanco,
    fontsize=18,
    fontweight='bold'
)
# =========================================================
# Loop principal
# =========================================================
try:
    while True:
        num_rx = rx_stream.recv(buffer, metadata)
        elapsed_time = time.time() - start_time

        if elapsed_time >= 20:
            titulo_global.set_text("Receiving")
        else:
            titulo_global.set_text("Sensing!!!")
            
        if num_rx > 0:
            x = buffer[0][:num_rx]

            if len(x) < nperseg:
                continue

            # PSD con Welch
            f_welch, psd = welch(
                x,
                fs=fs,
                window='hann',
                nperseg=nperseg,
                noverlap=noverlap,
                nfft=nfft,
                detrend=False,
                return_onesided=return_onesided,
                scaling='density'
            )

            # Reordenar para centrar el espectro en fc
            f_welch = np.fft.fftshift(f_welch)
            psd = np.fft.fftshift(psd)

            freq_axis = (f_welch + fc) / 1e6
            psd_db = 10 * np.log10(psd / 1e-3 + 1e-12)

            # -------------------------
            # Actualizar PSD
            # -------------------------
            line_psd.set_data(freq_axis, psd_db)
            ax1.set_xlim(freq_axis[0], freq_axis[-1])
            ax1.set_title(
                f"PSD en tiempo real - Welch | fc={fc/1e6:.3f} MHz | fs={fs/1e6:.2f} MS/s | gain={gain} dB",
                color=blanco
            )

            # -------------------------
            # Actualizar espectrograma
            # -------------------------
            spectrogram_data = np.roll(spectrogram_data, -1, axis=0)
            spectrogram_data[-1, :] = psd_db
            img.set_data(spectrogram_data)
            img.set_extent([freq_axis[0], freq_axis[-1], 0, historial])

            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.pause(0.001)

except KeyboardInterrupt:
    print("\n[INFO] Detenido por el usuario.")

finally:
    stop_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
    rx_stream.issue_stream_cmd(stop_cmd)
    plt.ioff()
    plt.show()