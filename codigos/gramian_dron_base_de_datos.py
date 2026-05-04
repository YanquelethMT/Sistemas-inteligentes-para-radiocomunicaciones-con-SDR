#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 13 14:22:56 2025
@author: ym-t
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, lfilter, welch
import scipy.signal as signal
import spkit as sp
from pyts.image import GramianAngularField
import os
import scipy.io
import time

# === Parámetros generales y configuración inicial ===
Fs = 84.5e6                 # Frecuencia de muestreo en Hz
Fc = 2.44225e9              # Frecuencia central de captura
bw = 15e6                   # Ancho de banda deseado

nfft = 1024
NFFT = int(nfft * 1)#1.042
segment_size = int(1e6)  
num_segments_aux = 1
q = int(3)                  # Factor de decimación
c_map='winter_r'
fraccion=50

# === Diseño del filtro FIR paso bajo para banda de 30 MHz ===
num_taps = 339
h = firwin(num_taps, cutoff=bw, fs=Fs)
h_1 = firwin(num_taps, cutoff=12.25e6, fs=Fs)
# === Ruta al archivo ===

ruta_base = '/home/ym-t/Descargas/DroneRFa/'

for j in range(1,2):  # 8 archivos, de 0000 a 0111
    bin_id = format(j, '04b')  # Número en binario de 4 bits
    nombre_archivo = f'T1011_S{bin_id}.mat'
    ruta_archivo = os.path.join(ruta_base, nombre_archivo)
    
    print(f'Cargando archivo: {ruta_archivo}')


    
    
    # === Carga de datos desde archivo .mat ===
    with h5py.File(ruta_archivo, 'r') as archivo:
        RF0_I = np.array(archivo['RF0_I']).flatten()
        RF0_Q = np.array(archivo['RF0_Q']).flatten()
        RF1_I = np.array(archivo['RF1_I']).flatten()
        RF1_Q = np.array(archivo['RF1_Q']).flatten()
    
    num_segments = len(RF0_I) // segment_size
    print(f"Total de segmentos: {num_segments}")
    
    # === Definición de frecuencias desplazadas para procesar ===
    f0_base = -27.25e6
    freq_offsets = [f0_base, f0_base + 30e6, f0_base + 60e6]
    
    # === Frecuencias base para graficar (ajuste final del eje x) === # ADAPTADO
    f_base_plot = [2.415e9, 2.445e9, 2.475e9]  # Inicio de cada rango
    f_base_plot_1 = [2.415e9, 2.445e9, 2.47225e9]  # Inicio de cada rango
    
    # === Procesamiento por segmentos ===
    for i in range(num_segments_aux):
    
        print("[INFO] Tomando un descanso de 60 segundos...")
        time.sleep(3)
        start = i * segment_size
        end = start + segment_size
    
        seg_I = RF0_I[start:end]
        seg_Q = RF0_Q[start:end]
        seg_complex = seg_I + 1j * seg_Q
         



        X=seg_complex
        # Aseguramos que X sea un vector de números complejos
        X_real = [complex(x.real, 0) for x in X]  # Convertir la parte real en complejos
        X_imag = [complex(0, x.imag) for x in X]  # Convertir la parte imaginaria en complejos
        
        """calculo de psd sobre cada parte"""
        Pxx, freqs = plt.psd(X, NFFT=NFFT, Fs=Fs, Fc=Fc)
        plt.show()
    
        Pxx_real, freqs_real = plt.psd(X_real, NFFT=NFFT, Fs=Fs, Fc=Fc)
        plt.show()
    
        Pxx_imag, freqs_imag = plt.psd(X_imag, NFFT=NFFT, Fs=Fs, Fc=Fc)
        plt.show()
        
        # Graficar la PSD de la parte real
        Pxx_total, freqs_total = plt.psd(X, NFFT=NFFT, Fs=Fs, Fc=Fc)
        plt.close()
    
        recorte = len(Pxx_real) // fraccion
        Pxx_real = Pxx_real[recorte : -recorte]
        Pxx_real = Pxx_real[:len(Pxx_real)//2]
        
        recorte = len(freqs_real) // fraccion
        freqs_real = freqs_real[recorte : -recorte]
        freqs_real = freqs_real[:len(freqs_real)//2]
        
        recorte = len(Pxx_imag) // fraccion
        Pxx_imag = Pxx_imag[recorte : -recorte]
        Pxx_imag = Pxx_imag[:len(Pxx_imag)//2]
    
        recorte = len(freqs_imag) // fraccion
        freqs_imag = freqs_imag[recorte : -recorte]
        freqs_imag = freqs_imag[:len(freqs_imag)//2]
              
        
        
        
        # ===== GAF ‑ SUMMATION =====
        gaf_sum_real = GramianAngularField(method='summation')
        X_gaf_sum_real = gaf_sum_real.fit_transform([Pxx_real])
        
        plt.figure(figsize=(4, 4))
        plt.imshow(X_gaf_sum_real[0], cmap='winter_r', aspect='auto')
        plt.title('GAF real– Summation')
        plt.axis('off')
        plt.show()
        
        gaf_sum_imag = GramianAngularField(method='summation')
        X_gaf_sum_imag = gaf_sum_imag.fit_transform([Pxx_imag])
        
        plt.figure(figsize=(4, 4))
        plt.imshow(X_gaf_sum_imag[0], cmap='winter_r', aspect='auto')
        plt.title('GAF imag– Summation')
        plt.axis('off')
        plt.show()
        
       
      
        

        
    print("[INFO] Tomando un descanso de 60 segundos...")
    time.sleep(60)