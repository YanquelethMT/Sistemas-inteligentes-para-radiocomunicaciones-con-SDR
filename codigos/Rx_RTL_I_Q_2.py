import numpy as np
import matplotlib.pyplot as plt
from rtlsdr import RtlSdr
from scipy.signal import firwin, lfilter, decimate, welch

# Inicializa SDR para LTE (850 MHz por ejemplo)
sdr = RtlSdr()
sdr.sample_rate = 3.2e6
sdr.center_freq = 915e6  # LTE banda 5 o 26 (depende del país)
sdr.gain = 'auto'

# Captura muestras
samples = sdr.read_samples(512*1024)
sdr.close()

# Filtro pasa bajos para una portadora (~300 kHz)
nyq_rate = sdr.sample_rate / 2.0
bw = 1e6
taps = firwin(numtaps=101, cutoff=bw/nyq_rate)
filtered = lfilter(taps, 1.0, samples)

# Decimación
decimation_factor = 10
decimated = decimate(filtered, decimation_factor)
fs_dec = sdr.sample_rate / decimation_factor

# ---------- PSD ----------
f, Pxx = welch(decimated, fs=fs_dec, window='hann', nperseg=1024, noverlap=512, return_onesided=True)
plt.figure(figsize=(10, 4))
plt.semilogy(f / 1e3, Pxx)
plt.title('Densidad Espectral de Potencia (LTE ~850 MHz)')
plt.xlabel('Frecuencia [kHz]')
plt.ylabel('PSD [V²/Hz]')
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------- Plano IQ ----------
plt.figure(figsize=(6, 6))
plt.scatter(np.real(decimated), np.imag(decimated), s=1, alpha=0.6)
plt.title('Plano IQ - Señal LTE (~850 MHz)')
plt.xlabel('Parte Real (I)')
plt.ylabel('Parte Imaginaria (Q)')
plt.grid(True)
plt.axis('equal')
plt.tight_layout()
plt.show()
