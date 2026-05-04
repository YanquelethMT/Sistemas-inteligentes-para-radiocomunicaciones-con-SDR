#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TDW multinivel + AMR + comparación de coeficientes
"""

import numpy as np
import matplotlib.pyplot as plt
import pywt

# =========================================================
# 1. Señal de prueba: 2 pulsos cuadrados + ruido
# =========================================================
np.random.seed(10)

fs = 2047  # Hz
t = np.linspace(0, 1, fs, endpoint=False)

x = np.zeros_like(t)

# Pulso 1
t1_inicio = 0.20
t1_fin    = 0.30
x[(t >= t1_inicio) & (t < t1_fin)] = 70

# Pulso 2
t2_inicio = 0.60
t2_fin    = 0.75
x[(t >= t2_inicio) & (t < t2_fin)] = 100

# Ruido gaussiano
ruido = 20 * np.random.randn(len(t))
x = x + ruido

# =========================================================
# 2. Parámetros wavelet
# =========================================================
wavelet = 'db1'
nivel_descomp = 3

# Verificación de nivel máximo permitido
nivel_max = pywt.dwt_max_level(data_len=len(x), filter_len=pywt.Wavelet(wavelet).dec_len)
nivel_descomp = min(nivel_descomp, nivel_max)

print(f"Wavelet seleccionada: {wavelet}")
print(f"Nivel máximo permitido: {nivel_max}")
print(f"Nivel de descomposición usado: {nivel_descomp}")

# =========================================================
# 3. Señal original
# =========================================================
plt.figure(figsize=(12, 4))
plt.plot(t, x, lw=1.3)
plt.title('Señal original: 2 pulsos cuadrados + ruido')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.grid(True)
plt.tight_layout()
plt.show()

# =========================================================
# 4. TDW multinivel
# coeffs = [cA_N, cD_N, cD_(N-1), ..., cD_1]
# =========================================================
coeffs = pywt.wavedec(x, wavelet, level=nivel_descomp)

cA_N = coeffs[0]
detalles = coeffs[1:]  # [cD_N, cD_(N-1), ..., cD_1]

# =========================================================
# 5. Mostrar todos los coeficientes de la TDW
# =========================================================
num_subplots = len(coeffs) + 1

plt.figure(figsize=(12, 2.2 * num_subplots))

# Señal original
plt.subplot(num_subplots, 1, 1)
plt.plot(x)
plt.title('Señal original')
plt.xlabel('Muestra')
plt.ylabel('Amplitud')
plt.grid(True)

# Aproximación final
plt.subplot(num_subplots, 1, 2)
plt.plot(cA_N)
plt.title(f'TDW - Coeficientes de aproximación cA{nivel_descomp}')
plt.xlabel('Índice')
plt.ylabel('Coef.')
plt.grid(True)

# Detalles
for i, cD in enumerate(detalles, start=1):
    nivel_actual = nivel_descomp - i + 1
    plt.subplot(num_subplots, 1, i + 2)
    plt.plot(cD)
    plt.title(f'TDW - Coeficientes de detalle cD{nivel_actual}')
    plt.xlabel('Índice')
    plt.ylabel('Coef.')
    plt.grid(True)

plt.tight_layout()
plt.show()

# =========================================================
# 6. Análisis multiresolución (AMR)
# Reconstrucción de componentes por nivel
# =========================================================
# Aproximación en el nivel escogido
compA = [np.zeros_like(c) for c in coeffs]
compA[0] = coeffs[0]
A_N = pywt.waverec(compA, wavelet)
A_N = A_N[:len(x)]

# Detalles reconstruidos de cada nivel
detalles_reconstruidos = []
for i in range(1, nivel_descomp + 1):
    comp = [np.zeros_like(c) for c in coeffs]   # todos en cero
    comp[i] = coeffs[i]                         # activar solo ese detalle
    Di = pywt.waverec(comp, wavelet)
    detalles_reconstruidos.append(Di[:len(x)])

# =========================================================
# 7. Mostrar AMR
# =========================================================
plt.figure(figsize=(12, 2.2 * (nivel_descomp + 2)))

plt.subplot(nivel_descomp + 2, 1, 1)
plt.plot(x)
plt.title('Señal original')
plt.xlabel('Muestra')
plt.ylabel('Amplitud')
plt.grid(True)

plt.subplot(nivel_descomp + 2, 1, 2)
plt.plot(A_N)
plt.title(f'AMR - Aproximación reconstruida A{nivel_descomp}')
plt.xlabel('Muestra')
plt.ylabel('Amplitud')
plt.grid(True)

for i, Di in enumerate(detalles_reconstruidos, start=1):
    nivel_actual = nivel_descomp - i + 1
    plt.subplot(nivel_descomp + 2, 1, i + 2)
    plt.plot(Di)
    plt.title(f'AMR - Detalle reconstruido D{nivel_actual}')
    plt.xlabel('Muestra')
    plt.ylabel('Amplitud')
    plt.grid(True)

plt.tight_layout()
plt.show()

# =========================================================
# 8. Coeficientes del AMR y comparación contra TDW
# =========================================================
print("\n================ COMPARACIÓN TDW vs AMR ================\n")
print(f"cA{nivel_descomp} obtenido por TDW (wavedec): tamaño = {len(cA_N)}")

for i, cD in enumerate(detalles, start=1):
    nivel_actual = nivel_descomp - i + 1
    print(f"cD{nivel_actual} obtenido por TDW (wavedec): tamaño = {len(cD)}")

print("\nInterpretación:")
print("- La TDW multinivel entrega directamente los coeficientes [cA_N, cD_N, ..., cD_1].")
print("- El AMR usa esos mismos coeficientes para reconstruir A_N y cada D_i en el dominio temporal.")
print("- Es decir: AMR no genera coeficientes distintos; usa los coeficientes de la TDW para separar contribuciones por escala.")

# =========================================================
# 9. Visualización puntual de coeficientes escogidos
# =========================================================
plt.figure(figsize=(12, 4))
plt.stem(cA_N, basefmt=' ')
plt.title(f'TDW - Coeficientes de aproximación cA{nivel_descomp}')
plt.xlabel('Índice')
plt.ylabel('Valor')
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 2.2 * nivel_descomp))
for i, cD in enumerate(detalles, start=1):
    nivel_actual = nivel_descomp - i + 1
    plt.subplot(nivel_descomp, 1, i)
    plt.plot(cD)
    plt.title(f'TDW - Coeficientes de detalle cD{nivel_actual}')
    plt.xlabel('Índice')
    plt.ylabel('Valor')
    plt.grid(True)

plt.tight_layout()
plt.show()

# =========================================================
# 7.1 Verificación de reconstrucción AMR
# =========================================================

# Suma de componentes
x_rec = A_N.copy()
for Di in detalles_reconstruidos:
    x_rec += Di

# Gráfica comparativa
plt.figure(figsize=(12, 6))

plt.plot(t, x, color='blue', linewidth=2.5, label='Señal original')
plt.plot(t, x_rec, color='red', linewidth=2.5, linestyle='--', label='Reconstrucción AMR')

plt.title('Verificación: x(t) vs A_N + ΣD_i')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.legend()
plt.grid(True)

plt.show()

# Error
error = x - x_rec

plt.figure(figsize=(12, 4))
plt.plot(t, error, color='black')
plt.title('Error de reconstrucción')
plt.xlabel('Tiempo [s]')
plt.ylabel('Error')
plt.grid(True)

plt.show()

# Error numérico
print(f"Error máximo: {np.max(np.abs(error)):.6f}")