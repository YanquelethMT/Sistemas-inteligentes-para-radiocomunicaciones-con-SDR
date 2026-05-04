"""
Created on Mon May 19 13:24:53 2025
@author: ym-t
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch
from rtlsdr import RtlSdr
import pywt

# Inicializamos el SDR
sdr = RtlSdr()
sdr.sample_rate = 2.4e6       # 2.4 MHz
sdr.center_freq = 100e6       # 100 MHz
sdr.gain = 'auto'

# Capturamos muestras
samples = sdr.read_samples(256*1024)
sdr.close()

# Welch: espectro completo (bidireccional) para señal IQ
nperseg = 1024
noverlap = 512
f, Pxx = welch(samples, fs=sdr.sample_rate, window='hann', nperseg=nperseg, noverlap=noverlap, return_onesided=False)

# Reacomodamos el espectro para centrarlo
f = np.fft.fftshift(f)
Pxx = np.fft.fftshift(Pxx)

# Evitar log10 de valores <= 0
Pxx = np.where(Pxx <= 0, 1e-12, Pxx)
Pxx_db = 10 * np.log10(Pxx)

# Frecuencia centrada en MHz
f_mhz = f / 1e6 + sdr.center_freq / 1e6

# Escalas para la CWT
scales = np.arange(0.2, 100,0.2)

# CWT con wavelet Morlet

coeficientes, frecuencias = pywt.cwt(Pxx_db, scales, 'morl')

# Módulo de coeficientes
modulo = (coeficientes)



# Figura con dos subplots verticales
plt.figure(figsize=(12, 10))

# Primer subplot: espectro en dB
plt.subplot(2, 1, 1)
plt.plot(f_mhz, Pxx_db)
plt.title("Densidad espectral de potencia (dB)")
plt.xlabel("Frecuencia [MHz]")
plt.ylabel("Potencia [dB]")
plt.grid(True)

# Segundo subplot: CWT con curva de máximo módulo
plt.subplot(2, 1, 2)
plt.imshow(modulo, extent=[f_mhz[0], f_mhz[-1], scales[-1], scales[0]], cmap='viridis_r', aspect='auto')
plt.gca().invert_yaxis()
plt.title("Transformada Continua de Wavelet (CWT) con curva de máximo")
plt.xlabel("Frecuencia [MHz]")
plt.ylabel("Escala")
# plt.colorbar(label='|Coeficientes|')
plt.legend()

plt.tight_layout()
plt.show()

# Trazar la fila 80 de los coeficientes (escala 81 porque scales[80] = 81)
aux = modulo[20]  # escala 81
plt.figure()
plt.plot(f_mhz, pow(np.abs(aux), 2))
plt.title("Coeficientes Wavelet en escala 81")
plt.xlabel("Frecuencia [MHz]")
plt.ylabel("|Coeficiente|")
plt.ylim(0, 30)


plt.grid(True)
plt.show()
