#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 12:39:22 2026

@author: ymt
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# Función DFH / HFD (basada en tu código)
# =========================================================
def HFD(serie, Kmax):
    try:
        N = len(serie)

        X = np.empty([N, Kmax, Kmax])
        X[:] = np.nan

        for k in range(1, Kmax + 1):
            for m in range(1, k + 1):
                limit = math.floor((N - m) / k)
                j = 1
                for i in range(m, (m + (limit * k)) + 1, k):
                    X[j - 1, k - 1, m - 1] = serie[i - 1]
                    j += 1

        L = np.zeros(Kmax)

        for k in range(1, Kmax + 1):
            Lm = np.zeros(k, dtype=float)

            for m in range(1, k + 1):
                if math.floor((N - m) / k) == 0:
                    R = math.inf
                else:
                    R = (N - 1) / (math.floor((N - m) / k) * k)

                r1 = list(X[:, k - 1, m - 1])
                aux = np.nan_to_num(r1)
                aux = aux[np.where(aux != 0)]

                for i in range(1, len(aux)):
                    Lm[m - 1] += abs(aux[i] - aux[i - 1])

                if (np.isnan(R) or np.isnan(Lm[m - 1]) or np.isnan(k) or
                    np.isinf(R) or np.isinf(Lm[m - 1]) or np.isinf(k)):
                    return 2
                else:
                    Lm[m - 1] = (R * Lm[m - 1]) / k

            L[k - 1] = sum(Lm) / k

        Xk = np.ones(Kmax, dtype=float)
        for i in range(1, Kmax + 1):
            Xk[i - 1] = i

        Xk = 1 / Xk

        LX = np.ones(len(Xk), dtype=float)
        for i in range(1, len(Xk) + 1):
            LX[i - 1] = math.log(Xk[i - 1])

        LL = np.ones(len(L), dtype=float)
        for i in range(1, len(L) + 1):
            if L[i - 1] == 0:
                LL[i - 1] = math.inf
            else:
                LL[i - 1] = math.log(L[i - 1])

        try:
            aux = np.polyfit(LX, LL, 1)
            return aux[0]
        except:
            return np.nan

    except (RuntimeError, TypeError, NameError, RuntimeWarning):
        return 2


# =========================================================
# Función SampEn (basada en tu código)
# =========================================================
def sampen(L):
    N = len(L)
    m = 2
    r = 0.1 * np.std(L)

    # Plantillas de longitud m
    xmi = np.array([L[i:i + m] for i in range(N - m)])
    xmj = np.array([L[i:i + m] for i in range(N - m + 1)])

    # Cálculo de B
    B = np.sum([
        np.sum(np.abs(xmii - xmj).max(axis=1) <= r) - 1
        for xmii in xmi
    ])

    # Cálculo de A para m+1
    m += 1
    xm = np.array([L[i:i + m] for i in range(N - m + 1)])

    A = np.sum([
        np.sum(np.abs(xmi - xm).max(axis=1) <= r) - 1
        for xmi in xm
    ])

    try:
        return -np.log10(A / B)
    except:
        return np.nan



# =========================================================
# Generación de la señal
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
ruido = 0 * np.random.randn(len(t))
x = x + ruido

# =========================================================
# Cálculo de métricas
# =========================================================
Kmax = 10

dfh_val = HFD(x, Kmax)
sampen_val = sampen(x)

print(f"DFH (Higuchi): {dfh_val:.6f}")
print(f"SampEn:        {sampen_val:.6f}")

# =========================================================
# Gráfica
# =========================================================
plt.figure(figsize=(12, 5))
plt.plot(t, x, linewidth=1)
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")
plt.title("Señal con dos pulsos y ruido gaussiano")
plt.grid(True)
plt.tight_layout()
plt.show()