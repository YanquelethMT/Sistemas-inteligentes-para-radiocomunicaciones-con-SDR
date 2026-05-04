#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 21 16:47:00 2026

@author: ymt
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
from scipy.signal import welch
from scipy.signal.windows import hann

if __name__ == "__main__":

    # ==========================
    # PARÁMETROS
    # ==========================
    Fc = 2.442e9              # Frecuencia central
    fs = 84.5e6               # Frecuencia de muestreo
    NFFT = 2048
    t_captura = 0.05
    muestras_por_trama = int(fs * t_captura)

    nperseg = 2048
    noverlap = int(0.5 * nperseg)
    step = nperseg - noverlap
    window = hann(nperseg)

    ruta_base = '/home/ymt/Descargas/DroneRFa/'

    # ==========================
    # LOOP SOBRE ARCHIVOS
    # ==========================
    for i in range(2, 8):
        bin_id = format(i, '04b')
        nombre_archivo = f'T{bin_id}_D00_S0000.mat'
        ruta_archivo = os.path.join(ruta_base, nombre_archivo)

        print(f"\nProcesando archivo: {nombre_archivo}")

        if not os.path.isfile(ruta_archivo):
            print("  [!] Archivo no encontrado")
            continue

        # ==========================
        # LEER ARCHIVO
        # ==========================
        with h5py.File(ruta_archivo, 'r') as archivo:
            RF0_I = np.array(archivo['RF0_I']).flatten()
            RF0_Q = np.array(archivo['RF0_Q']).flatten()

        num_bloques = len(RF0_I) // muestras_por_trama
        num_bloques = min(num_bloques, 5)

        print(f"  → Bloques a procesar: {num_bloques}")

        # ==========================
        # LOOP SOBRE BLOQUES
        # ==========================
        for b in range(num_bloques):

            inicio = b * muestras_por_trama
            fin = inicio + muestras_por_trama

            seg_T = RF0_I[inicio:fin] + 1j * RF0_Q[inicio:fin]

            print(f"    Procesando bloque {b+1}/{num_bloques}")

            # ============================================================
            # PSD POR FFT SEGMENTADA (WELCH MANUAL, como SDR)
            # ============================================================
            psd_fft_segments = []

            num_segments = (len(seg_T) - noverlap) // step

            for k in range(num_segments):
                start = k * step
                end = start + nperseg

                segment = seg_T[start:end]

                if len(segment) < nperseg:
                    break

                segment = segment * window

                Xf = np.fft.fftshift(np.fft.fft(segment, NFFT))
                psd = (np.abs(Xf) ** 2) / (fs * np.sum(window**2))
                if k%400==0:
                    psd_fft_segments.append(psd)

            PSD_fft_manual = np.mean(psd_fft_segments, axis=0)
            PSD_fft_manual = 10 * np.log10(PSD_fft_manual/1e-3 + 1e-12)

            f_fft = np.fft.fftshift(np.fft.fftfreq(NFFT, 1/fs))
            f_fft_mhz = (f_fft + Fc) / 1e9

            # ============================================================
            # PSD POR WELCH (SCIPY)
            # ============================================================
            f_welch, PSD_welch = welch(
                seg_T,
                fs=fs,
                window='hann',
                nperseg=nperseg,
                noverlap=noverlap,
                nfft=NFFT,
                return_onesided=False,
                scaling='density'
            )

            PSD_welch = np.fft.fftshift(PSD_welch)
            f_welch = np.fft.fftshift(f_welch)

            PSD_welch = 10 * np.log10(PSD_welch/1e-3 + 1e-12)
            f_welch_mhz = (f_welch + Fc) / 1e9

            # ============================================================
            # GRÁFICAS
            # ============================================================
            plt.figure(figsize=(11, 7))

            plt.subplot(2, 1, 1)
            plt.plot(f_fft_mhz, PSD_fft_manual)
            plt.xlabel('Frecuencia [GHz]')
            plt.ylabel('PSD [dBm]')
            plt.grid(True)

            plt.subplot(2, 1, 2)
            plt.plot(f_welch_mhz, PSD_welch)
            plt.xlabel('Frecuencia [GHz]')
            plt.ylabel('PSD [dBm]')
            plt.grid(True)

            plt.tight_layout()
            plt.show()
