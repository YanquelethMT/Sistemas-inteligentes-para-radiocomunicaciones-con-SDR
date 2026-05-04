#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 14:56:50 2026

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
# fc = 315e6             # Frecuencia central [Hz] LLAVES
# fs = 50e6              # Frecuencia de muestreo [Hz]LLAVES

# fc = 91e6             # Frecuencia central [Hz] FM
# fs = 2e6              # Frecuencia de muestreo [Hz] FM

# fc = 5.8e9             # Frecuencia central [Hz] DRONE
# fs = 50e6              # Frecuencia de muestreo [Hz]DRONE

fc = 2.426e9             # Frecuencia central [Hz] FM
fs=40e6 

bw = fs                # Ancho de banda RX [Hz]
gain = 20             # Ganancia RX [dB]
fft_size = 2048        # Tamaño base para Welch
tiempo = 0.001         # Tiempo de bloque para visualización [s]
muestras = int(fs * tiempo)
device_args = "type=b200"
antenna = "TX/RX"
channel = 0

# Parámetros del espectrograma
historial = 150        # Número de columnas temporales visibles

# Parámetros del cambio de modo
tiempo_sensing = 100    # [s] tiempo antes de entrar a Receiving
t_aux = 1              # [s] duración de la adquisición auxiliar
muestras_aux = int(t_aux * fs)

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

fig = plt.figure(figsize=(15, 6))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.04], wspace=0.25)

ax1 = fig.add_subplot(gs[0, 0])   # PSD
ax2 = fig.add_subplot(gs[0, 1])   # Espectrograma
cax = fig.add_subplot(gs[0, 2])   # Colorbar

fig.patch.set_facecolor('black')
ax1.set_facecolor('black')
ax2.set_facecolor('black')
cax.set_facecolor('black')

turquesa = '#00FFF7'
gris_grid = '#444444'
blanco = '#FFFFFF'

# =========================================================
# Título global dinámico
# =========================================================
titulo_global = fig.suptitle(
    "Sensing!!!",
    color=blanco,
    fontsize=18,
    fontweight='bold',
    y=0.98
)

# =========================================================
# Subplot 1: PSD
# =========================================================
line_psd, = ax1.plot([], [], color=turquesa, linewidth=1.5)

ax1.set_title("PSD en tiempo real - Ettus B210 (Welch)", color=blanco, pad=12)
ax1.set_xlabel("Frecuencia [MHz]", color=blanco)
ax1.set_ylabel("Potencia [dBm/Hz]", color=blanco)
ax1.set_ylim(-80, -30)
ax1.grid(True, color=gris_grid, linestyle='--', linewidth=0.5, alpha=0.7)
ax1.tick_params(axis='x', colors=blanco)
ax1.tick_params(axis='y', colors=blanco)

for spine in ax1.spines.values():
    spine.set_color(gris_grid)

# =========================================================
# Subplot 2: Espectrograma girado 90°
# filas   -> frecuencia
# columnas-> tiempo
# =========================================================
spectrogram_data = np.full((nfft, historial), -180.0)

freq_axis_temp = (np.fft.fftshift(np.fft.fftfreq(nfft, d=1/fs)) + fc) / 1e6
time_axis_temp = np.arange(historial)

img = ax2.imshow(
    spectrogram_data,
    aspect='auto',
    cmap='hot',
    origin='lower',
    extent=[time_axis_temp[0], time_axis_temp[-1], freq_axis_temp[0], freq_axis_temp[-1]],
    vmin=-100,
    vmax=-60
)

ax2.set_title("Espectrograma en tiempo real", color=blanco, pad=12)
ax2.set_xlabel("Tiempo", color=blanco)
ax2.set_ylabel("Frecuencia [MHz]", color=blanco)
ax2.tick_params(axis='x', colors=blanco)
ax2.tick_params(axis='y', colors=blanco)

for spine in ax2.spines.values():
    spine.set_color(gris_grid)

cbar = fig.colorbar(img, cax=cax)
cbar.set_label("Potencia [dBm/Hz]", color=blanco)
cbar.ax.yaxis.set_tick_params(color=blanco)
plt.setp(cbar.ax.get_yticklabels(), color=blanco)

# Reservar espacio para el título global
fig.subplots_adjust(top=0.86, left=0.06, right=0.95, bottom=0.12)

# =========================================================
# Variables de control de estados
# =========================================================
start_time = time.time()
estado = "sensing"
captura_aux_hecha = False

# =========================================================
# Función: captura auxiliar de 5 segundos
# =========================================================
def capturar_muestras_aux(rx_stream, metadata, total_samples, block_size):
    """
    Captura total_samples muestras complejas desde el stream continuo,
    leyendo por bloques de tamaño block_size.
    """
    print("")
    print("")
    print(f"[INFO] Iniciando captura auxiliar de {total_samples} muestras...")
    print("")


    data_aux = np.empty(total_samples, dtype=np.complex64)
    temp_buffer = np.zeros((1, block_size), dtype=np.complex64)

    idx = 0
    while idx < total_samples:
        num_rx = rx_stream.recv(temp_buffer, metadata)

        if num_rx <= 0:
            continue

        n_copy = min(num_rx, total_samples - idx)
        data_aux[idx:idx+n_copy] = temp_buffer[0][:n_copy]
        idx += n_copy
    print("")

    print("[INFO] Captura auxiliar finalizada.")
    print("")
    print("")

    return data_aux

# =========================================================
# Loop principal
# =========================================================
try:
    while True:

        elapsed_time = time.time() - start_time

        # -------------------------------------------------
        # Cambio a modo Receiving al cumplir 20 s
        # -------------------------------------------------
        if (elapsed_time >= tiempo_sensing) and (not captura_aux_hecha):
            estado = "receiving"
            titulo_global.set_text("Receiving")
            fig.canvas.draw()
            fig.canvas.flush_events()

            # Captura auxiliar grande
            muestras_capturadas = capturar_muestras_aux(
                rx_stream=rx_stream,
                metadata=metadata,
                total_samples=muestras_aux,
                block_size=muestras
            )

            print(f"[INFO] Se capturaron {len(muestras_capturadas)} muestras auxiliares.")

            # Aquí puedes guardar, procesar o inspeccionar muestras_capturadas
            # por ejemplo:
            np.save("captura_aux.npy", muestras_capturadas)

            captura_aux_hecha = True
            estado = "sensing"
            titulo_global.set_text("Sensing!!!")
            start_time = time.time()   # reiniciar ciclo de 20 s
            continue

        # -------------------------------------------------
        # Modo normal: PSD + espectrograma
        # -------------------------------------------------
        if estado == "sensing":
            titulo_global.set_text("Sensing!!!")

            num_rx = rx_stream.recv(buffer, metadata)

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
                    return_onesided=False,   # ← importante
                    scaling='density'
                )
                
                # Centrar espectro
                psd = np.fft.fftshift(psd)
                f_welch = np.fft.fftshift(f_welch)

              
                freq_axis = (f_welch + fc) / 1e6
                psd_db = 10 * np.log10(psd / 1e-3 + 1e-12)

                # -------------------------
                # Actualizar PSD
                # -------------------------
                line_psd.set_data(freq_axis, psd_db)
                ax1.set_xlim(freq_axis[0], freq_axis[-1])
                ax1.set_title(
                    f"PSD en tiempo real - Welch | fc={fc/1e6:.3f} MHz | fs={fs/1e6:.2f} MS/s | gain={gain} dB",
                    color=blanco,
                    pad=12
                )

                # -------------------------
                # Actualizar espectrograma
                # Cada nueva PSD entra como una nueva columna temporal
                # -------------------------
                spectrogram_data = np.roll(spectrogram_data, -1, axis=1)
                spectrogram_data[:, -1] = psd_db

                img.set_data(spectrogram_data)
                img.set_extent([0, historial - 1, freq_axis[0], freq_axis[-1]])

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