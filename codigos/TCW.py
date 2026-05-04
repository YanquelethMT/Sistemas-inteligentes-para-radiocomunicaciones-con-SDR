import numpy as np
import matplotlib.pyplot as plt
import pywt
# =========================================
# 1. Señal de prueba (2 pulsos cuadrados + ruido)
# =========================================
fs = 1000  # Hz
t = np.linspace(0, 1, fs, endpoint=False)

# Inicializar señal en cero
x = np.zeros_like(t)

# =========================================
# Pulsos cuadrados
# =========================================

# Pulso 1
t1_inicio = 0.2
t1_fin    = 0.3
x[(t >= t1_inicio) & (t < t1_fin)] = 70

# Pulso 2
t2_inicio = 0.6
t2_fin    = 0.75
x[(t >= t2_inicio) & (t < t2_fin)] = 100

# =========================================
# Ruido Gaussiano
# =========================================
ruido = 20 * np.random.randn(len(t))  # ajusta potencia aquí
x = x + ruido

# =========================================
# Parámetros de escala
# =========================================
escala_i = 1
escala_f = 20

# =========================================
# 2. Visualización de la señal
# =========================================
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 4))
plt.plot(t, x, label='Señal con pulsos + ruido')

# marcar regiones de pulsos (opcional pero útil)
plt.axvspan(t1_inicio, t1_fin, alpha=0.2, label='Pulso 1')
plt.axvspan(t2_inicio, t2_fin, alpha=0.2, label='Pulso 2')

plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.title('Señal: Pulsos cuadrados + ruido gaussiano')
plt.legend()
plt.grid()

plt.show()
# =========================================
# 3. Definición de escalas
# =========================================
scales = np.arange(escala_i, escala_f,0.1)

# # Wavelet (Morlet compleja)
# Wavelets COMPLEJAS (las más importantes para SDR / frecuencia)
# 'cmor1.0-1.0'
# 'cmor1.5-1.0'
# 'cmor2.0-1.0'
# 'cmor3.0-1.0'
# 'cmor1.5-0.5'
# 'shan'
# 'fbsp2-1.0-1.0'
# 'fbsp3-1.5-1.0'
#  Wavelets REALES (útiles para análisis estructural)
# 'mexh'
# 'morl'
# 'gaus1'
# 'gaus2'
# 'gaus3'
# 'gaus4'
# 'gaus5'
# 'gaus6'
# 'gaus7'
# 'gaus8'
# 'gaus9'
# 'gaus10'
wavelet = 'gaus1'

# =========================================
# 4. Transformada Continua Wavelet
# =========================================
coeffs, freqs = pywt.cwt(x, scales, wavelet)

# =========================================
# 5. Escalograma (|CWT|)
# =========================================
plt.figure(figsize=(10, 6))

plt.imshow(np.abs(coeffs),
           extent=[t[0], t[-1], scales[-1], scales[0]],
           aspect='auto',
           cmap='jet')

plt.colorbar(label='|CWT|')
plt.xlabel('Tiempo [s]')
plt.ylabel('Escala')
plt.title('Escalograma - Transformada Continua Wavelet')

plt.show()

# =========================================
# 6. Selección de una escala específica
# =========================================
escala_objetivo = 20  # puedes cambiar esto
idx = np.argmin(np.abs(scales - escala_objetivo))

cwt_escala = coeffs[idx, :]

# =========================================
# 7. Módulo y máximo
# =========================================
modulo = np.abs(cwt_escala)
max_val = np.max(modulo)
t_max = t[np.argmax(modulo)]

print(f"Escala seleccionada: {scales[idx]}")
print(f"Valor máximo del módulo: {max_val:.4f}")
print(f"Ocurre en t = {t_max:.4f} s")

# =========================================
# 8. Visualización de esa escala
# =========================================
plt.figure(figsize=(10, 4))

plt.plot(t, modulo, label='|CWT(a,b)|')
plt.axhline(max_val, linestyle='--', label='Máximo')

plt.xlabel('Tiempo [s]')
plt.ylabel('Magnitud')
plt.title(f'Módulo de la CWT en escala = {scales[idx]}')

plt.legend()
plt.grid()

plt.show()