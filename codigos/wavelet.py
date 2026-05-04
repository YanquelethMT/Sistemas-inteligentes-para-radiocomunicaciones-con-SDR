import numpy as np
import pywt
import matplotlib.pyplot as plt
from pywt import wavedec
    
import numpy as np
import pywt
import matplotlib.pyplot as plt



def wrcoef(X, coef_type, coeffs, wavename, level):
    N = np.array(X).size
    a, ds = coeffs[0], list(reversed(coeffs[1:]))

    if coef_type =='a':
        return pywt.upcoef('a', a, wavename, level=level)[:N]
    elif coef_type == 'd':
        return pywt.upcoef('d', ds[level-1], wavename, level=level)[:N]
    else:
        raise ValueError("Invalid coefficient type: {}".format(coef_type))
        
# Generamos una señal: seno + ruido
longi=4096
t = np.linspace(0, 1, longi)
signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.random.randn(longi)

# Aplicamos la DWT usando wavelet 'db4'
wavelet = 'db1'
cA, cD = pywt.dwt(signal, wavelet)

# Reconstrucción de la señal desde los coeficientes
reconstructed_signal = pywt.idwt(cA, cD, wavelet)

# Visualizamos
plt.figure(figsize=(12, 6))

plt.subplot(3, 1, 1)
plt.plot(signal)
plt.title("Señal Original (seno + ruido)")

plt.subplot(3, 1, 2)
plt.plot(cA)
plt.title("Coeficientes de Aproximación")

plt.subplot(3, 1, 3)
plt.plot(reconstructed_signal)
plt.title("Señal Reconstruida desde DWT")

plt.tight_layout()
plt.show()


# Señal original: seno + ruido
t = np.linspace(0, 1, longi)
signal = np.sin(2 * np.pi * 5 * t) + 0.5 * np.random.randn(longi)

# Parámetros de la DWT
wavelet = 'db2'
max_level = 5

# Descomposición multiresolución
coeffs = pywt.wavedec(signal, wavelet, level=max_level)

# Reconstrucción de la señal desde todos los coeficientes
reconstructed_signal = pywt.waverec(coeffs, wavelet)

# Visualización de la señal original, los detalles y la reconstrucción
plt.figure(figsize=(12, 10))

# Señal original
plt.subplot(len(coeffs)+2, 1, 1)
plt.plot(signal)
plt.title("Señal Original (seno + ruido)")

# Coeficientes (Aprox. y Detalles)
for i, coef in enumerate(coeffs):
    plt.subplot(len(coeffs)+2, 1, i+2)
    plt.plot(coef)
    if i == 0:
        plt.title(f"Aproximación final (Nivel {max_level})")
    else:
        plt.title(f"Detalle nivel {max_level - i + 1}")

# Señal reconstruida
plt.subplot(len(coeffs)+2, 1, len(coeffs)+2)
plt.plot(reconstructed_signal)
plt.title("🔁 Señal Reconstruida desde todos los niveles")

plt.tight_layout()
plt.show()

plt.figure()
plt.stem(coeffs[0])
plt.title("Coeficientes Originales")
plt.xlabel("Índice")
plt.ylabel("Valor")
plt.grid(True)
plt.show()

# Normalización entre 0 y 1
coeffs_rescaled = (coeffs[0] - np.min(coeffs[0])) / (np.max(coeffs[0]) - np.min(coeffs[0]))
plt.figure()
plt.stem(coeffs_rescaled)
plt.title("Coeficientes Normalizados entre 0 y 1")
plt.xlabel("Índice")
plt.ylabel("Valor Normalizado")
plt.grid(True)
plt.show()

# Normalización entre -1 y 1
max_abs = np.max(np.abs(coeffs[0]))
coeffs_scaled = coeffs[0] / max_abs
plt.figure()
plt.stem(coeffs_scaled)
plt.title("Coeficientes Normalizados entre -1 y 1")
plt.xlabel("Índice")
plt.ylabel("Valor Escalado")
plt.grid(True)
plt.show()

nivelwave=5


coeffs = wavedec(signal, wavelet, level=nivelwave)
cons = wrcoef(signal, 'a', coeffs, wavelet, nivelwave)
plt.figure()
plt.plot(signal, label='Señal Original')
plt.plot(cons, label=f'Reconstrucción Nivel {nivelwave}')
plt.title('Reconstrucción Wavelet')
plt.xlabel('Índice de Muestra')
plt.ylabel('Amplitud')
plt.legend()
plt.grid(True)
plt.show()
