#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 19 15:25:58 2025

@author: ym-t
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 19 13:24:53 2025
@author: ym-t
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from rtlsdr import RtlSdr

# Inicializamos el SDR
sdr = RtlSdr()
sdr.sample_rate = 3e6       # 2.4 MHz
sdr.center_freq = 91.3e6       # 100 MHz
sdr.gain = 'auto'

# Capturamos muestras
samples = sdr.read_samples(256*1024)
sdr.close()

# Visualización simple de una parte de la señal
plt.figure(figsize=(10, 3))
plt.plot(np.real(samples[:1024]))
plt.title('Señal original (parte real)')
plt.xlabel('Índice')
plt.ylabel('Amplitud')
plt.grid(True)
plt.tight_layout()
plt.show()

# Welch: espectro completo (bidireccional) para señal IQ
nperseg = 1024
noverlap = 512
f, Pxx = welch(samples, fs=sdr.sample_rate, window='hann', nperseg=nperseg, noverlap=noverlap, return_onesided=False)

# Reacomodamos el espectro para centrarlo
f = np.fft.fftshift(f)
Pxx = np.fft.fftshift(Pxx)

# Centrado en 100 MHz
f_mhz = f / 1e6 + sdr.center_freq / 1e6

# Graficamos la PSD
plt.figure(figsize=(10, 4))
plt.semilogy(f_mhz, Pxx)
plt.title('Densidad Espectral de Potencia usando Welch (centrado en 100 MHz)')
plt.xlabel('Frecuencia [MHz]')
plt.ylabel('PSD [V²/Hz]')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plano IQ
plt.figure(figsize=(6, 6))
plt.scatter(np.real(samples), np.imag(samples), s=1, alpha=0.7)
plt.title('Plano IQ')
plt.xlabel('Parte Real (I)')
plt.ylabel('Parte Imaginaria (Q)')
plt.grid(True)
plt.axis('equal')
plt.tight_layout()
plt.show()


from sklearn.cluster import KMeans

iq_data = np.column_stack((np.real(samples[::10]), np.imag(samples[::10])))

# Prueba con diferentes números de clusters
for k in [4, 16, 64]:
    kmeans = KMeans(n_clusters=k, random_state=0).fit(iq_data)
    inertia = kmeans.inertia_
    print(f'k={k}, inertia={inertia}')
