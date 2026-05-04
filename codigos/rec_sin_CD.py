#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 16:54:51 2026

@author: ymt
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AMR usando wrcoef + reconstrucción sin D1
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
    """
    N = len(X)
    a = coeffs[0]
    ds = list(reversed(coeffs[1:]))   # ds[0]=cD1, ds[1]=cD2, ...

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
plt.title(f'AMR - Aproximación reconstruida A{nivel} con wrcoef')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.grid(True)

for i, D_lev in enumerate(detalles, start=1):
    plt.subplot(nivel + 2, 1, i + 2)
    plt.plot(t, D_lev)
    plt.title(f'AMR - Detalle reconstruido D{i} con wrcoef')
    plt.xlabel('Tiempo [s]')
    plt.ylabel('Amplitud')
    plt.grid(True)

plt.tight_layout()
plt.show()

# =========================================================
# 7. Reconstrucción completa y reconstrucción sin D1
# =========================================================

# Reconstrucción completa: A_n + D1 + D2 + ... + Dn
x_rec_full = A_n.copy()
for D_lev in detalles:
    x_rec_full += D_lev

# Reconstrucción sin D1: A_n + D2 + D3 + ... + Dn
x_rec_sin_D1 = A_n.copy()
for i, D_lev in enumerate(detalles, start=1):
    if i != 1:   # excluir D1
        x_rec_sin_D1 += D_lev

# =========================================================
# 8. Comparaciones
# =========================================================

# Comparación: original vs A_n
plt.figure(figsize=(12, 6))
plt.plot(t, x, color='blue', linewidth=2.5, label='Señal original')
plt.plot(t, A_n, color='red', linewidth=2.5, label='A_n')
plt.title('Comparación entre señal original y aproximación A_n')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Comparación: original vs reconstrucción completa
plt.figure(figsize=(12, 6))
plt.plot(t, x, color='blue', linewidth=2.5, label='Señal original')
plt.plot(t, x_rec_full, color='red', linewidth=2.5, linestyle='--',
         label='Reconstrucción completa: A_n + ΣD_i')
plt.title('Comparación entre señal original y reconstrucción completa por AMR')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Comparación: original vs reconstrucción sin D1
plt.figure(figsize=(12, 6))
plt.plot(t, x, color='blue', linewidth=2.5, label='Señal original')
plt.plot(t, x_rec_sin_D1, color='red', linewidth=2.5, linestyle='--',
         label='Reconstrucción sin D1: A_n + D2 + ... + Dn')
plt.title('Reconstrucción parcial sin el detalle D1')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# =========================================================
# 9. Error de reconstrucción
# =========================================================
error_full = x - x_rec_full
error_sin_D1 = x - x_rec_sin_D1

plt.figure(figsize=(12, 5))
plt.plot(t, error_full, linewidth=1.8, label='Error reconstrucción completa')
plt.plot(t, error_sin_D1, linewidth=1.8, label='Error reconstrucción sin D1')
plt.title('Errores de reconstrucción')
plt.xlabel('Tiempo [s]')
plt.ylabel('Error')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print(f"Error máximo reconstrucción completa: {np.max(np.abs(error_full)):.6e}")
print(f"Error máximo reconstrucción sin D1:   {np.max(np.abs(error_sin_D1)):.6e}")

# =========================================================
# 10. Mostrar coeficientes asociados
# =========================================================
plt.figure(figsize=(12, 4))
plt.plot(coeffs[0])
plt.title(f'TDW - Coeficientes cA{nivel}')
plt.xlabel('Índice')
plt.ylabel('Valor')
plt.grid(True)
plt.tight_layout()
plt.show()

