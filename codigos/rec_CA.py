#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AMR usando wrcoef
"""

import numpy as np
import matplotlib.pyplot as plt
import pywt
from pywt import wavedec

# =========================================================
# 1. Señal de prueba: 2 pulsos cuadrados + ruido
# =========================================================
np.random.seed(10)

fs = 4096  # Hz
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
ruido = 10 * np.random.randn(len(t))
x = x + ruido

# =========================================================
# 2. Función wrcoef
# =========================================================
def wrcoef(X, coef_type, coeffs, wavename, level):
    """
    Reconstruye una aproximación o detalle específico usando los
    coeficientes de wavedec.
    
    Parámetros
    ----------
    X : array
        Señal original, solo se usa para ajustar la longitud final.
    coef_type : str
        'a' para aproximación, 'd' para detalle.
    coeffs : list
        Lista devuelta por pywt.wavedec -> [cA_N, cD_N, ..., cD_1]
    wavename : str
        Nombre de la wavelet.
    level : int
        Nivel a reconstruir.
    """
    N = len(X)
    a = coeffs[0]
    ds = list(reversed(coeffs[1:]))   # ahora ds[0]=cD1, ds[1]=cD2, ...

    if coef_type == 'a':
        rec = pywt.upcoef('a', a, wavename, level=level)
        return rec[:N]

    elif coef_type == 'd':
        rec = pywt.upcoef('d', ds[level - 1], wavename, level=level)
        return rec[:N]

    else:
        raise ValueError("coef_type debe ser 'a' o 'd'")

# =========================================================
# 3. Parámetros de descomposición
# =========================================================
wavelet = 'db1'
nivel = 5

nivel_max = pywt.dwt_max_level(data_len=len(x), filter_len=pywt.Wavelet(wavelet).dec_len)
nivel = min(nivel, nivel_max)

print(f"Wavelet: {wavelet}")
print(f"Nivel máximo permitido: {nivel_max}")
print(f"Nivel usado: {nivel}")

# =========================================================
# 4. Descomposición
# =========================================================
coeffs = wavedec(x, wavelet, level=nivel)

# =========================================================
# 5. Reconstrucción AMR con wrcoef
# =========================================================
A_n = wrcoef(x, 'a', coeffs, wavelet, nivel)

detalles = []
for lev in range(1, nivel + 1):
    D_lev = wrcoef(x, 'd', coeffs, wavelet, lev)
    detalles.append(D_lev)

# =========================================================
# 6. Mostrar señal original y componentes reconstruidos
# =========================================================
plt.figure(figsize=(12, 2.4 * (nivel + 2)))

plt.subplot(nivel + 2, 1, 1)
plt.plot(t, x)
plt.title('Señal original')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.grid(True)

plt.subplot(nivel + 2, 1, 2)
plt.plot(t, A_n)
plt.title(f'Aproximación reconstruida A{nivel} con wrcoef')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.grid(True)

for i, D_lev in enumerate(detalles, start=1):
    plt.subplot(nivel + 2, 1, i + 2)
    plt.plot(t, D_lev)
    plt.title(f'Detalle reconstruido D{i} con wrcoef')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud')
    plt.grid(True)

plt.tight_layout()
plt.show()

# =========================================================
# 7. Verificación de reconstrucción parcial
# =========================================================
x_rec = A_n.copy()


plt.figure(figsize=(12, 6))

plt.plot(t, x, color='blue', linewidth=2.5, label='Señal original')
plt.plot(t, x_rec, color='red', linewidth=2.5, label='A_n')

plt.title('Comparación entre señal original y reconstrucción por AMR')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.legend()
plt.grid(True)


for D_lev in detalles:
    x_rec += D_lev

plt.figure(figsize=(12, 6))

plt.plot(t, x, label='Señal original')
plt.plot(t, x_rec, '--', label='Suma A_n + ΣD_i')
plt.title('Comparación entre señal original y reconstrucción por AMR')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.legend()
plt.grid(True)


plt.tight_layout()
plt.show()

# =========================================================
# 8. Mostrar coeficientes asociados
# =========================================================
plt.figure(figsize=(12, 4))
plt.plot(coeffs[0])
plt.title(f'Coeficientes cA{nivel}')
plt.xlabel('Índice')
plt.ylabel('Valor')
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 2.2 * nivel))
for i in range(1, len(coeffs)):
    nivel_actual = nivel - i + 1
    plt.subplot(nivel, 1, i)
    plt.plot(coeffs[i])
    plt.title(f'Coeficientes cD{nivel_actual}')
    plt.xlabel('Índice')
    plt.ylabel('Valor')
    plt.grid(True)

plt.tight_layout()
plt.show()