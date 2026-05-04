#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 15:11:48 2026

@author: ymt
"""


import numpy as np
import matplotlib.pyplot as plt
from hilbertcurve.hilbertcurve import HilbertCurve
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import spkit as sp
from scipy.signal import welch
from pyts.image import GramianAngularField

from matplotlib.gridspec import GridSpec



# =============================
#       CURVA DE HILBERT
# =============================
def obtener_coords_hilbert(p):
    """
    Regresa las coordenadas (x,y) del recorrido Hilbert
    para una curva de orden p.
    """
    N = 2
    hil = HilbertCurve(p, N)
    total = (2**p)**2
    coords = hil.points_from_distances(list(range(total)))
    return coords, 2**p, total



def generar_hilbert(signal_1d, p=7):
    """
    Convierte una señal 1D en una imagen Hilbert (2^p x 2^p).
    Regresa únicamente la matriz numérica Hilbert.
    """
    coords, side, total = obtener_coords_hilbert(p)

    # Normalizamos a 0–255
    vmin, vmax = np.min(signal_1d), np.max(signal_1d)
    if vmax - vmin == 0:
        norm = np.zeros_like(signal_1d)
    else:
        norm = 255 * (signal_1d - vmin) / (vmax - vmin)

    # Repetimos la señal para cubrir toda la curva si es necesario
    
    rep = int(np.ceil(total / len(norm)))
    padded = np.tile(norm, rep)[:total]

    # Construcción de imagen Hilbert
    img = np.zeros((side, side), dtype=np.uint8)
    for i, (r, c) in enumerate(coords):
        img[r, c] = int(padded[i])

    return img

# ==================================================================
#            EJEMPLO DE USO: aplicar ambas técnicas a X
# ==================================================================

if __name__ == "__main__":
    
    # ==========================
    # Constantes
    # ==========================
    Fc = 2.442e9
    fs = 84.5e6
    NFFT = 2048
    t_captura = 0.05
    muestras_por_trama = int(fs * t_captura)
    nperseg = 2048
    porcen = 0.50
    noverlap = int(nperseg * porcen)
    c_map = 'winter_r'
    c_map = 'RdPu'
    # c_map = 'gray_r'

    ruta_base = '/home/ymt/Descargas/DroneRFa/'
    
    # Lista de nombres de drones en el orden binario
    nombres_drones = [
        "background", "phantom 3", "phantom 4", "matrice 200", "air 2s", "mini 3 pro",
        "inspire 2", "mavic pro", "mini 2", "mavic 3", "matrice 300",
        "phantom 4 pro RTK", "matrice 30 T", "avata", "chino", "drone_desconocido"
    ]
    
    # ==========================
    # Loop sobre archivos
    # ==========================
    for i in range(8, 10):  # cambia a range(1, 16) si quieres los 15 archivos
        bin_id = format(i, '04b')  # Convierte a binario de 4 bits
        nombre_archivo = f'T{bin_id}_D00_S0000.mat'
        print(f"\nProcesando archivo: {nombre_archivo}")
        ruta_archivo = os.path.join(ruta_base, nombre_archivo)
    
        if not os.path.isfile(ruta_archivo):
            print(f"[!] Archivo no encontrado: {ruta_archivo}")
            continue
    
        # Leer archivo .mat
        with h5py.File(ruta_archivo, 'r') as archivo:
            RF0_I = np.array(archivo['RF0_I']).flatten()
            RF0_Q = np.array(archivo['RF0_Q']).flatten()
    
        # Cuántos bloques caben completos en la señal
        num_bloques = len(RF0_I) // muestras_por_trama
        print(f"  → Total de bloques: {num_bloques}")
        num_bloques=6
        # Loop sobre bloques
        for b in range(num_bloques):
            inicio = b * muestras_por_trama
            fin = inicio + muestras_por_trama
    
            seg_I = RF0_I[inicio:fin]
            seg_Q = RF0_Q[inicio:fin]
            seg_T = seg_I + 1j * seg_Q
    
            print(f"    Procesando bloque {b+1}/{num_bloques}...")
            
            #PSD
            
            #PSD
            f_w, Pxx = welch(seg_T, fs=fs, nperseg=NFFT,
                             return_onesided=False, scaling='density')
            f_w = np.fft.fftshift(f_w) + Fc
            Pxx = np.fft.fftshift(Pxx)
            Pxx = 10 * np.log10(Pxx/1e-3 + 1e-12)
            
            # FRFT
            seg_T_frft = sp.frft(seg_T, alpha=1)
            
            f_w2, Pxx2 = welch(seg_T_frft, fs=fs, nperseg=NFFT,
                               return_onesided=False, scaling='density')
            f_w2 = np.fft.fftshift(f_w2) + Fc
            Pxx2 = np.fft.fftshift(Pxx2)
            Pxx2 = 10 * np.log10(Pxx2/1e-3 + 1e-12)
            t_welch = np.linspace(0, t_captura, len(Pxx2))
            
            img_hil = generar_hilbert(Pxx2, p=8)
            
            # =========================
            # Figura única con 3 plots
            # =========================
            fig = plt.figure(figsize=(14, 6), dpi=120)
            gs = GridSpec(2, 2, width_ratios=[1, 1.05], height_ratios=[1, 1], wspace=0.28, hspace=0.30)
            
            # PSD izquierda arriba
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.plot(f_w/1e9, Pxx)
            ax1.set_title("PSD")
            ax1.set_xlabel("Frequency [GHz]")
            ax1.set_ylabel("Power [dBm]")
            ax1.grid(True)
            
            # Envolvente izquierda abajo
            ax2 = fig.add_subplot(gs[1, 0])
            ax2.plot(t_welch, Pxx2)
            ax2.set_title("Envolvente")
            ax2.set_xlabel("Time [s]")
            ax2.set_ylabel("Amplitude [dBm]")
            ax2.grid(True)
            
            # Hilbert derecha completa
            ax3 = fig.add_subplot(gs[:, 1])
            ax3.imshow(img_hil, cmap=c_map, aspect='auto')
            ax3.set_title("Curva de Hilbert")
            ax3.axis('off')
            
            fig.suptitle(f"Bloque {b+1}", fontsize=14)
            plt.tight_layout()
            plt.show()
