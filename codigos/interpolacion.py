#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 20 07:50:21 2025

@author: ym-t
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from pykrige.ok import OrdinaryKriging


n = 15
X = np.random.uniform(0, 10, n)
Y = np.random.uniform(0, 10, n)
Z = np.random.uniform(-100, -30, n)

grid_x = np.linspace(0, 10, 200)
grid_y = np.linspace(0, 10, 200)
grid_x, grid_y = np.meshgrid(grid_x, grid_y)

def idw(x, y, z, xi, yi, power=2):
    interpolated = np.zeros_like(xi)
    for i in range(xi.shape[0]):
        for j in range(xi.shape[1]):
            d = np.sqrt((x - xi[i, j])**2 + (y - yi[i, j])**2)
            if np.any(d == 0):
                interpolated[i, j] = z[d == 0][0]
            else:
                w = 1 / d**power
                interpolated[i, j] = np.sum(w * z) / np.sum(w)
    return interpolated

idw_result = idw(X, Y, Z, grid_x, grid_y)

try:
    OK = OrdinaryKriging(
        X, Y, Z,
        variogram_model='spherical',  
        variogram_parameters={'sill': np.var(Z), 'range': 5, 'nugget': 0.1},
        verbose=False,
        enable_plotting=False,
        coordinates_type='euclidean'
    )
    
    kriging_result, ss = OK.execute('grid', grid_x[0, :], grid_y[:, 0])
    
    if np.any(np.isnan(kriging_result)):
        mean_val = np.nanmean(kriging_result)
        kriging_result = np.nan_to_num(kriging_result, nan=mean_val)
except Exception as e:
    print(f"Error en Kriging: {e}")
    kriging_result = np.full(grid_x.shape, np.mean(Z))  

fig, axs = plt.subplots(1, 3, figsize=(18, 6))

sc0 = axs[0].scatter(X, Y, c=Z, cmap='jet', s=80, edgecolors='black')
axs[0].set_title("Puntos originales (dBm)")
axs[0].set_xlabel("X")
axs[0].set_ylabel("Y")
plt.colorbar(sc0, ax=axs[0])

c1 = axs[1].imshow(idw_result,
                  extent=(0, 10, 0, 10),
                  origin='lower',
                  cmap='jet')
axs[1].scatter(X, Y, c='white', edgecolors='black', s=30)
axs[1].set_title("Interpolación IDW (dBm)")
plt.colorbar(c1, ax=axs[1])

c2 = axs[2].imshow(kriging_result,
                  extent=(0, 10, 0, 10),
                  origin='lower',
                  cmap='jet')
axs[2].scatter(X, Y, c='white', edgecolors='black', s=30)
axs[2].set_title("Interpolación Kriging (dBm)")
plt.colorbar(c2, ax=axs[2])

plt.tight_layout()
plt.show()