#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# %%
import os
import warnings
import time


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy import signal
from scipy.signal import welch

import SoapySDR
from SoapySDR import SOAPY_SDR_RX, SOAPY_SDR_CF32, SOAPY_SDR_TIMEOUT


# =============================
# Ajustes
# =============================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["SOAPY_SDR_LOG_LEVEL"] = "0"
warnings.simplefilter("ignore", category=ImportWarning)
warnings.simplefilter("ignore", category=DeprecationWarning)

# =============================
# Parámetros
# =============================
n_font = 8

fs = 20e6          # estable (sube luego si quieres)
bw = fs
fc_inicio = 0.290e9
fc = fc_inicio + bw/2


gain = 50          # sube un poco para ver más contraste (prueba 40-60)

NFFT = 512
nperseg = NFFT
noverlap = int(nperseg * 0.50)

t_captura = 0.01
muestras_por_trama = int(fs * t_captura)

c_map = "winter_r"

# Contraste (más agresivo)
P_LOW = 5     # percentil bajo
P_HIGH = 98   # percentil alto

# Normalización para que se note el contraste:
# True = quita el piso por frecuencia (MUY recomendado)
NORMALIZAR_POR_FREC = True

# =============================
# Detectar e inicializar SDR
# =============================
results = SoapySDR.Device.enumerate()
SDRs = []
for r in results:
    desc = str(r).lower()
    if any(tag in desc for tag in ("hackrf", "rtlsdr", "lime", "blade")):
        try:
            SDRs.append(SoapySDR.Device(r))
        except Exception:
            pass

if not SDRs:
    try:
        SDRs.append(SoapySDR.Device("driver=lime"))
    except Exception:
        pass

if not SDRs:
    raise RuntimeError(
        "No se encontró ningún dispositivo SDR conectado.\n"
        "Tip: verifica en terminal `SoapySDRUtil --find`."
    )

print(f"Detectados: {len(SDRs)} SDR(s)")

rxStreams = []
for idx, sdr in enumerate(SDRs):
    ch = 0
    sdr.setSampleRate(SOAPY_SDR_RX, ch, fs)
    sdr.setFrequency(SOAPY_SDR_RX, ch, fc + idx * bw)

    try:
        sdr.setBandwidth(SOAPY_SDR_RX, ch, bw)
    except Exception:
        pass
    try:
        sdr.setGain(SOAPY_SDR_RX, ch, gain)
    except Exception:
        pass

    stream = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [ch])
    sdr.activateStream(stream)
    rxStreams.append((sdr, stream))

num_sdrs = len(rxStreams)

# =============================
# Preparar gráficos
# =============================
plt.ion()
top_fig = plt.figure(figsize=(11, 8))
gs_main = GridSpec(2, 1, figure=top_fig, hspace=0.35)

ax_psd = top_fig.add_subplot(gs_main[0, 0])

gs_spec = GridSpecFromSubplotSpec(num_sdrs, 1, subplot_spec=gs_main[1, 0], hspace=0.35)
axes_spec = [top_fig.add_subplot(gs_spec[r, 0]) for r in range(num_sdrs)]

FPS = 5
dt = 1.0 / FPS



# =============================
# Loop de captura
# =============================
try:
    while True:
        ax_psd.cla()
        for a in axes_spec:
            a.cla()

        for idx, (sdr, rxStream) in enumerate(rxStreams):
            # ----- Captura -----
            buff = np.empty(muestras_por_trama, dtype=np.complex64)
            total_recv = 0
            while total_recv < muestras_por_trama:
                sr = sdr.readStream(rxStream, [buff[total_recv:]], muestras_por_trama - total_recv)
                if sr.ret > 0:
                    total_recv += sr.ret
                elif sr.ret == SOAPY_SDR_TIMEOUT:
                    break
                else:
                    break

            if total_recv == 0:
                continue

            x = buff[:total_recv]

            # ----- PSD -----
            f, Pxx = welch(x, fs=fs, nperseg=NFFT, return_onesided=False, scaling="density")
            f = np.fft.fftshift(f) + (fc + idx * bw)
            Pxx_db = 10 * np.log10(np.fft.fftshift(Pxx) / 1e-3 + 1e-12)

            ax_psd.plot(f / 1e6, Pxx_db, label=f"SDR {idx}", alpha=0.85)

            # ----- Espectrograma -----
            f1, t1, Sxx = signal.spectrogram(
                x, fs=fs, nperseg=nperseg, noverlap=noverlap,
                return_onesided=False, scaling="density", mode="psd"
            )

            # Sxx: [freq, time]
            Sxx_db = 10 * np.log10(np.fft.fftshift(Sxx, axes=0) + 1e-12)

            # ✅ normaliza por frecuencia: resta la mediana de cada fila (cada frecuencia)
            if NORMALIZAR_POR_FREC:
                fila_med = np.median(Sxx_db, axis=1, keepdims=True)
                Sxx_vis = Sxx_db - fila_med
                # ahora la escala es "dB relativos al piso" (más visible)
                zlabel = "Potencia [dB rel.]"
            else:
                Sxx_vis = Sxx_db
                zlabel = "Potencia [dB]"

            f1 = np.fft.fftshift(f1) + (fc + idx * bw)

            # ✅ percentiles más agresivos
            vmin = np.percentile(Sxx_vis, P_LOW)
            vmax = np.percentile(Sxx_vis, P_HIGH)

            im = axes_spec[idx].imshow(
                Sxx_vis,
                aspect="auto",
                cmap=c_map,
                origin="lower",
                extent=[t1.min(), t1.max(), f1.min() / 1e6, f1.max() / 1e6],
                vmin=vmin,
                vmax=vmax,
            )
            axes_spec[idx].set_title(
    f"Espectrograma SDR {idx}  (vmin={vmin:.1f}, vmax={vmax:.1f})",
    fontsize=n_font
)
            
            


            axes_spec[idx].set_ylabel("Frecuencia [MHz]", fontsize=n_font)
            if idx == num_sdrs - 1:
                axes_spec[idx].set_xlabel("Tiempo [s]", fontsize=n_font)
            else:
                axes_spec[idx].set_xticklabels([])

            

        # ----- Configurar PSD -----
        ax_psd.set_title("PSD (todas las SDR)", fontsize=n_font)
        ax_psd.set_xlabel("Frecuencia [MHz]", fontsize=n_font)
        ax_psd.set_ylabel("PSD [dB]", fontsize=n_font)
        ax_psd.grid(True, alpha=0.3)
        ax_psd.legend(fontsize=n_font - 2)

        plt.pause(0.001)
        time.sleep(dt)

except KeyboardInterrupt:
    print("Interrumpido por el usuario")

finally:
    for sdr, rxStream in rxStreams:
        try:
            sdr.deactivateStream(rxStream)
            sdr.closeStream(rxStream)
        except Exception:
            pass
    print("Captura finalizada")
