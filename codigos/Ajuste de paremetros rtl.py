from rtlsdr import RtlSdr
import numpy as np
import matplotlib.pyplot as plt

sdr = RtlSdr()
sdr.sample_rate = 3.2e6
sdr.center_freq = 91.3e6   # 100 MHz
sdr.gain = 80#'auto'

nfft=4096
samples1 = sdr.read_samples(1000*1024)
sdr.close()

# Graficar espectro
plt.figure(figsize=(10, 6))
plt.psd(samples1, NFFT=nfft, Fs=sdr.sample_rate/1e6, Fc=sdr.center_freq/1e6)
plt.xlabel('Frecuencia [MHz]')
plt.ylabel('dB')
plt.title('Espectro RTL-SDR')
plt.show()







# sdr = RtlSdr()
# sdr.sample_rate = 2.4e6
# sdr.center_freq = 100e6   # 100 MHz
# sdr.gain = 80#'auto'
# nfft=1024
# samples2 = sdr.read_samples(1000*1024)
# sdr.close()


# # Graficar espectro
# plt.figure(figsize=(10, 6))
# plt.psd(samples2, NFFT=nfft, Fs=sdr.sample_rate/1e6, Fc=sdr.center_freq/1e6)
# plt.xlabel('Frecuencia [MHz]')
# plt.ylabel('dB')
# plt.title('Espectro RTL-SDR')
# plt.show()


# samples=samples1+samples2


# plt.figure(figsize=(10, 6))
# plt.psd(samples, NFFT=nfft, Fs=sdr.sample_rate/1e6, Fc=sdr.center_freq/1e6)
# plt.xlabel('Frecuencia [MHz]')
# plt.ylabel('dB')
# plt.title('Espectro RTL-SDR')
# plt.show()

# # Guardar muestras en CSV como números complejos (formato string)
# with open('samples.csv', 'w') as f:
#     f.write('samples\n')  # encabezado
#     for s in samples:
#         f.write(f'{s}\n')

# print("✅ Muestras guardadas en samples.csv como números complejos")


# samples = np.loadtxt('samples.csv', dtype=complex, skiprows=1)

# print(f"✅ {len(samples)} muestras cargadas")

# # Configuración del muestreo (ajusta si es necesario)

# # Graficar PSD
# plt.figure(figsize=(10, 6))
# plt.psd(samples, NFFT=nfft, Fs=sdr.sample_rate/1e6, Fc=sdr.center_freq/1e6)
# plt.xlabel('Frecuencia [MHz]')
# plt.ylabel('dB')
# plt.title('Espectro desde CSV')
# plt.grid(True)
# plt.show()