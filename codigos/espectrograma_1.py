

"""
Created on Fri May 16 12:34:45 2025

@author: ym-t
"""

import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Constantes
Fc = 2.442e9              # Frecuencia central (Hz)
Fs = 84.5e6
Fs_1 = int(100e6)         # Frecuencia de muestreo alternativa
NFFT = 2048
segment_size = int(7e6)   # Tamaño del segmento a analizar
c_map = 'viridis'
ruta_base = '/home/ym-t/Descargas/DroneRFa/'
output_dir = 'espectrogramas'

# Crear carpeta de salida si no existe
os.makedirs(output_dir, exist_ok=True)

# Lista de nombres de drones en el orden binario
nombres_drones = [
    "background", "phantom 3", "phantom 4 pro", "matrice 200","matrice 100", 
    "air 2s", "mini 3 pro", "inspire 2", "mavic pro", "mini 2", "mavic 3", 
    "matrice 300", "phantom 4 pro RTK", "matrice 30 T", "avata", "chino","Matrice 600 pro"]

# Loop de archivos
for i in range(1, 2):  # Modifica el rango según los archivos que quieras procesar
    bin_id = format(i, '04b')  # Convierte a binario de 4 bits
    nombre_archivo = f'T{bin_id}_D00_S0000.mat' 
    print(f"[INFO] Procesando: {nombre_archivo}")
    
    ruta_archivo = os.path.join(ruta_base, nombre_archivo)
    
    if not os.path.isfile(ruta_archivo):
        print(f"[!] Archivo no encontrado: {ruta_archivo}")
        continue

    # Cargar datos IQ
    with h5py.File(ruta_archivo, 'r') as archivo:
        RF0_I = np.array(archivo['RF0_I']).flatten()
        RF0_Q = np.array(archivo['RF0_Q']).flatten()
        RF1_I = np.array(archivo['RF1_I']).flatten()
        RF1_Q = np.array(archivo['RF1_Q']).flatten()
        
        ############### 2.4
    # Seleccionar segmento y construir señal compleja
    seg0_I = RF0_I[0:segment_size]
    seg0_Q = RF0_Q[0:segment_size]
    seg0_T = seg0_I + 1j * seg0_Q

    # Parámetros para espectrograma
    fs = 84.5e6  # frecuencia de muestreo intermedia para el espectrograma
    nperseg = 1024
    noverlap = 512
    
    # Calcular espectrograma para señal COMPLEJA
    f, t, Sxx = signal.spectrogram(
        seg0_T,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        return_onesided=False,  # Necesario para señal compleja
        scaling='density',
        mode='psd'
    )
    
    # Centrar el espectro y ajustar al eje real de frecuencias
    # f = np.fft.fftshift(f)   # Ajustar eje de frecuencia al rango real
    f=f+fs/2
    f=f+2.4e9
    Sxx = np.fft.fftshift(Sxx, axes=0)
    f = np.fft.fftshift(f, axes=0)
    
    # Convertir a decibeles (dB)
    Sxx_dB = 10 * np.log10(np.abs(Sxx/1e-3))
    
    # Visualizar espectrograma
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t, f / 1e9, Sxx_dB, shading='auto', cmap=c_map)
    plt.ylabel('Frecuencia [GHz]')
    plt.xlabel('Tiempo [s]')
    plt.title(f'Espectrograma - {nombres_drones[i]} 2.4 GHz')
    plt.colorbar(label='Potencia [dB]')
    plt.tight_layout()
    
    # plt.savefig(os.path.join(output_dir, aux), dpi=300)
    # plt.close()
    plt.show()