#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reproducción visual simple de un .npy grabado con captura_iq_simple_npy.py.

Pensado para Spyder y para archivos de varios segundos:
- Abre el .npy con mmap (sin cargar todo a RAM).
- Recorre la captura por ventanas.
- Muestra solo la PSD de la ventana actual con Welch.
- Avanza en el tiempo como una reproducción visual.

Ideal cuando captura_s = 6 s o más.
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

# =========================================================
# Parámetros de usuario
# =========================================================
archivo_npy = "captura_aux.npy"   # Archivo generado por captura_iq_simple_npy.py
fs = 2e6                                  # Debe coincidir con el fs de captura
fc = 91.3e6                                 # Frecuencia central usada en captura

# Reproducción visual
ventana_ms = 50.0                          # Duración de cada ventana analizada
solapamiento = 0.0                       # 0.0 = sin solapamiento
velocidad_reproduccion = 100000               # 1.0 = tiempo real, 5.0 = 5 veces más rápido
actualizar_cada_n = 1                      # 1 = cada ventana, 2 = una sí y una no

# Welch
nperseg = 2048
noverlap = 1024
nfft = 2048

# =========================================================
# Carga segura
# =========================================================
print("[INFO] Abriendo archivo...")
iq = np.load(archivo_npy, mmap_mode="r")

if iq.ndim != 1:
    raise ValueError("Se esperaba un arreglo 1D de muestras IQ complejas.")

if not np.iscomplexobj(iq):
    raise TypeError("El archivo .npy no contiene muestras complejas.")

N = len(iq)
duracion_s = N / fs
tam_mb = os.path.getsize(archivo_npy) / (1024**2)

print(f"[INFO] Archivo: {archivo_npy}")
print(f"[INFO] dtype: {iq.dtype}")
print(f"[INFO] shape: {iq.shape}")
print(f"[INFO] Total de muestras: {N:,}")
print(f"[INFO] Duración: {duracion_s:.3f} s")
print(f"[INFO] Tamaño: {tam_mb:.2f} MB")
print(f"[INFO] Primeras 20 muestras: {np.asarray(iq[:20])}")

# =========================================================
# Parámetros derivados
# =========================================================
muestras_ventana = int(round(fs * ventana_ms * 1e-3))
muestras_ventana = max(muestras_ventana, nperseg)

paso = int(round(muestras_ventana * (1.0 - solapamiento)))
paso = max(1, paso)

num_frames = 1 + max(0, (N - muestras_ventana) // paso)

if num_frames <= 0:
    raise RuntimeError("No hay suficientes muestras para procesar con la ventana elegida.")

print(f"[INFO] Ventana: {ventana_ms:.3f} ms ({muestras_ventana:,} muestras)")
print(f"[INFO] Paso: {paso:,} muestras")
print(f"[INFO] Frames estimados: {num_frames:,}")

# =========================================================
# Primera PSD para inicializar la gráfica
# =========================================================
x0 = np.asarray(iq[:muestras_ventana], dtype=np.complex64)

f_welch, psd = welch(
    x0,
    fs=fs,
    window="hann",
    nperseg=min(nperseg, len(x0)),
    noverlap=min(noverlap, max(0, min(nperseg, len(x0)) - 1)),
    nfft=nfft,
    detrend=False,
    return_onesided=False,
    scaling="density"
)

f_welch = np.fft.fftshift(f_welch)
psd = np.fft.fftshift(psd)

freq_axis_mhz = (f_welch + fc) / 1e6
psd_db = 10 * np.log10(psd + 1e-12)

# =========================================================
# Gráfica interactiva
# =========================================================
plt.ion()
plt.close("all")

fig, ax = plt.subplots(figsize=(12, 6))
line_psd, = ax.plot(freq_axis_mhz, psd_db, linewidth=1.2)

ax.set_title("PSD por Welch - reproducción visual del archivo IQ")
ax.set_xlabel("Frecuencia [MHz]")
ax.set_ylabel("PSD [dB]")
ax.grid(True)
ax.set_xlim(freq_axis_mhz[0], freq_axis_mhz[-1])

texto_estado = fig.text(
    0.5, 0.97, "Preparando...",
    ha="center", va="top",
    fontsize=11, fontweight="bold"
)

plt.tight_layout(rect=[0, 0, 1, 0.94])

# =========================================================
# Reproducción visual
# =========================================================
print("[INFO] Iniciando reproducción visual...")

try:
    for frame_idx, start in enumerate(range(0, N - muestras_ventana + 1, paso)):
        if frame_idx % actualizar_cada_n != 0:
            continue

        stop = start + muestras_ventana
        x = np.asarray(iq[start:stop], dtype=np.complex64)

        f_welch, psd = welch(
            x,
            fs=fs,
            window="hann",
            nperseg=min(nperseg, len(x)),
            noverlap=min(noverlap, max(0, min(nperseg, len(x)) - 1)),
            nfft=nfft,
            detrend=False,
            return_onesided=False,
            scaling="density"
        )

        f_welch = np.fft.fftshift(f_welch)
        psd = np.fft.fftshift(psd)

        freq_axis_mhz = (f_welch + fc) / 1e6
        psd_db = 10 * np.log10(psd + 1e-12)

        line_psd.set_data(freq_axis_mhz, psd_db)

        ymin = np.percentile(psd_db, 5) - 3
        ymax = np.percentile(psd_db, 95) + 3
        if np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin:
            ax.set_ylim(ymin, ymax)

        t_ini = start / fs
        t_fin = stop / fs
        texto_estado.set_text(
            f"Ventana: {t_ini:.3f} s a {t_fin:.3f} s  |  "
            f"Duración total: {duracion_s:.3f} s  |  "
            f"Frame {frame_idx + 1}/{num_frames}"
        )

        fig.canvas.draw()
        fig.canvas.flush_events()

        dt_senal = (paso / fs) / max(velocidad_reproduccion, 1e-9)
        plt.pause(0.001)

    print("[INFO] Reproducción visual terminada.")

except KeyboardInterrupt:
    print("[INFO] Reproducción visual interrumpida por el usuario.")

finally:
    plt.ioff()
    plt.show()
