import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import spkit as sp
from scipy.signal import welch
from pyts.image import GramianAngularField

# ==========================
# Constantes
# ==========================
Fc = 2.442e9
fs = 84.5e6
NFFT = 2048
t_captura = 0.005
muestras_por_trama = int(fs * t_captura)
nperseg = 2048
porcen = 0.50
noverlap = int(nperseg * porcen)
c_map = 'winter_r'
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
for i in range(5, 6):  # cambia a range(1, 16) si quieres los 15 archivos
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
    num_bloques=3
    # Loop sobre bloques
    for b in range(num_bloques):
        inicio = b * muestras_por_trama
        fin = inicio + muestras_por_trama

        seg_I = RF0_I[inicio:fin]
        seg_Q = RF0_Q[inicio:fin]
        seg_T = seg_I + 1j * seg_Q

        print(f"    Procesando bloque {b+1}/{num_bloques}...")

        # FRFT
        seg_T_frft = sp.frft(seg_T, alpha=1)

        # === Espectrograma ===
        f, t, Sxx = signal.spectrogram(
            seg_T, fs=fs, nperseg=nperseg, noverlap=noverlap,
            return_onesided=False, scaling='density', mode='psd'
        )
        
        Sxx = np.fft.fftshift(Sxx, axes=0)
        f = np.fft.fftshift(f, axes=0)
        f = f + Fc   # Espectrograma en RF absoluta
        Sxx = 10 * np.log10(Sxx/1e-3 + 1e-12)
        extent = [t.min(), t.max(), f.min()/1e9, f.max()/1e9]
        
        # === Welch en banda base ===
        f_w, Pxx = welch(
            seg_T,
            fs=fs,
            nperseg=512,
            return_onesided=False,
            scaling='density'
        )
        
        f_w = np.fft.fftshift(f_w)
        # Pxx = np.fft.fftshift(Pxx)
        Pxx = 10 * np.log10(Pxx/1e-3 + 1e-12)
        
        # === Figura I/Q ===
        t_s = np.linspace(t.min(), t.max(), len(seg_Q))
        
        plt.figure(figsize=(8, 4))
        plt.plot(t_s, seg_Q, label='Q Component (Imag)', color='r')
        plt.plot(t_s, seg_I, label='I Component (Real)', color='b')
        plt.xlabel("Time [s]")
        plt.ylabel("Amplitude [V]")
        plt.legend(loc='upper right')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
        # === Figura PSD en banda base ===
        plt.figure(figsize=(8, 4))
        plt.plot(f_w / 1e6, Pxx)
        plt.title(f"Bloque {b+1} - PSD Welch en banda base")
        plt.xlabel("Frecuencia [MHz]")
        plt.ylabel("PSD [dBm/Hz]")
        plt.grid(True)
        plt.tight_layout()
        plt.show()