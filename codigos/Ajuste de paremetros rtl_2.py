
from rtlsdr import RtlSdr
import time

# Inicializa el SDR
sdr = RtlSdr()
print("SDR conectado")

# Ajusta los parámetros del SDR
sdr.sample_rate = 2.4e6     # Tasa de muestreo en Hz (2.4 MHz)
sdr.center_freq = 100.1e6   # Frecuencia central en Hz (100.1 MHz)
sdr.gain = 'auto'           # Ganancia automática

print(f"Frecuencia central: {sdr.center_freq / 1e6} MHz")
print(f"Tasa de muestreo: {sdr.sample_rate / 1e6} MHz")
print(f"Ganancia: {sdr.gain}")

time.sleep(1)

# Cierra el dispositivo cuando termines
sdr.close()
print("SDR desconectado")
