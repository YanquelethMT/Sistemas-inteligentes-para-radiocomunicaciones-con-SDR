import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from rtlsdr import RtlSdr

Fs=3e6
NFFT=1024
Fc=91.3e6
# Configurar RTL-SDR
sdr = RtlSdr()
sdr.sample_rate = Fs          # Hz
sdr.center_freq = Fc         # Hz (por ejemplo, FM)
sdr.gain = 'auto'

# Parámetros de adquisición
fs = sdr.sample_rate
n_blocks = 100                  # Número de bloques
block_size = 1024*10                # Muestras por bloque
nperseg = 1024                   # Tamaño de ventana para espectrograma
noverlap = nperseg // 2          # Superposición
c_map = 'viridis'                # Colormap

# Captura por bloques
print("Capturando muestras...")
all_samples = []
for _ in range(n_blocks):
    samples = sdr.read_samples(block_size)
    all_samples.append(samples)
sdr.close()
print("Captura finalizada.")

# Concatenar muestras en un solo arreglo
samples_concat = np.concatenate(all_samples)


for i in range(n_blocks):
    plt.psd(all_samples[i],NFFT=NFFT, Fs=Fs,Fc=Fc)
plt.show()

# Calcular espectrograma
f, t, Sxx = signal.spectrogram(
    samples_concat,
    fs=fs,
    nperseg=nperseg,
    noverlap=noverlap,
    return_onesided=False,   # Señal compleja
    scaling='density',
    mode='psd'
)



# Ajustar eje de frecuencias y centrar espectro
f = np.fft.fftshift(f)
f = f #+ fs / 2
f = f + sdr.center_freq       # Ajustar eje al valor real de frecuencia

Sxx = np.fft.fftshift(Sxx, axes=0)

# Convertir a dB
Sxx_dB = 10 * np.log10(np.abs(Sxx / 1e-3))

# Graficar espectrograma
plt.figure(figsize=(10, 6))
plt.pcolormesh(t, f / 1e6, Sxx_dB, shading='auto', cmap=c_map)
plt.ylabel('Frecuencia [MHz]')
plt.xlabel('Tiempo [s]')
plt.title('Espectrograma RTL-SDR por bloques')
plt.colorbar(label='Potencia [dB]')
plt.tight_layout()
plt.show()
