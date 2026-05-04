#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 15:20:07 2026

@author: ymt
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import uhd

# =========================================================
# Parámetros de usuario
# =========================================================
archivo_npy = "captura_aux.npy"
device_args = "type=b200"
channel = 0

# =========================================================
# Parámetros de transmisión
# =========================================================
tx_fc = 91.3e6       # Hz FM
tx_fs = 2e6          # Hz FM

# tx_fc = 295e6       # Hz LLAVES
# tx_fs = 50e6          # Hz LLAVES

# tx_fc = 5.8e9       # Hz DRONE
# tx_fs = 50e6          # Hz DRONE


tx_bw = tx_fs        # Hz
tx_gain = 100         # dB, AJUSTAR CON CUIDADO
tx_ant = "TX/RX"

# Reproducción
repeticiones = 10     # cuántas veces se transmite el archivo
bloque_envio = 8192  # muestras por bloque
bloque_envio = 1048576 #DRON
peak_objetivo = 0.9  # amplitud pico máxima

# =========================================================
# Carga del archivo IQ
# =========================================================
print("[INFO] Abriendo archivo IQ...")
iq = np.load(archivo_npy, mmap_mode="r")

if iq.ndim != 1:
    raise ValueError("Se esperaba un arreglo 1D de muestras IQ complejas.")

if not np.iscomplexobj(iq):
    raise TypeError("El archivo .npy no contiene muestras complejas.")

N = len(iq)
duracion = N / tx_fs
tam_mb = os.path.getsize(archivo_npy) / (1024**2)

print(f"[INFO] Archivo: {archivo_npy}")
print(f"[INFO] dtype original: {iq.dtype}")
print(f"[INFO] muestras: {N:,}")
print(f"[INFO] duración: {duracion:.3f} s")
print(f"[INFO] tamaño: {tam_mb:.2f} MB")

# Convertir a complex64
iq_tx = np.asarray(iq, dtype=np.complex64)

# =========================================================
# Escalado de amplitud
# =========================================================
max_abs = np.max(np.abs(iq_tx))
if max_abs <= 0:
    raise ValueError("La señal tiene amplitud nula.")

iq_tx = iq_tx / max_abs
iq_tx = iq_tx * peak_objetivo

if not np.all(np.isfinite(iq_tx.real)) or not np.all(np.isfinite(iq_tx.imag)):
    raise ValueError("La señal contiene NaN o Inf.")

print(f"[INFO] Señal normalizada. Pico final = {np.max(np.abs(iq_tx)):.3f}")

# =========================================================
# Inicialización del USRP
# =========================================================
print("[INFO] Abriendo USRP...")
usrp = uhd.usrp.MultiUSRP(device_args)

usrp.set_tx_rate(tx_fs, channel)
usrp.set_tx_freq(tx_fc, channel)
usrp.set_tx_gain(tx_gain, channel)

try:
    usrp.set_tx_bandwidth(tx_bw, channel)
except Exception as e:
    print(f"[WARN] No se pudo configurar el ancho de banda TX: {e}")

try:
    usrp.set_tx_antenna(tx_ant, channel)
except Exception as e:
    print(f"[WARN] No se pudo configurar la antena TX: {e}")

print(f"[INFO] tx_fs real: {usrp.get_tx_rate(channel)/1e6:.6f} MS/s")
print(f"[INFO] tx_fc real: {usrp.get_tx_freq(channel)/1e6:.6f} MHz")
print(f"[INFO] tx_bw real: {usrp.get_tx_bandwidth(channel)/1e6:.6f} MHz")
print(f"[INFO] tx_gain real: {usrp.get_tx_gain(channel):.2f} dB")

# =========================================================
# Configuración del TX streamer
# =========================================================
st_args = uhd.usrp.StreamArgs("fc32", "sc16")
st_args.channels = [channel]

tx_stream = usrp.get_tx_stream(st_args)
tx_md = uhd.types.TXMetadata()

# =========================================================
# Función de transmisión
# =========================================================
def transmitir_burst(tx_stream, tx_md, sig, bloque=8192):
    total = len(sig)
    enviados = 0
    primer_bloque = True

    while enviados < total:
        fin = min(enviados + bloque, total)
        chunk = sig[enviados:fin]

        tx_md.start_of_burst = primer_bloque
        tx_md.end_of_burst = (fin >= total)
        tx_md.has_time_spec = False

        n = tx_stream.send(chunk, tx_md)

        if n <= 0:
            print("[WARN] No se enviaron muestras en este bloque.")
            continue

        if n != len(chunk):
            print(f"[WARN] Se enviaron {n} de {len(chunk)} muestras en este bloque.")

        enviados += n
        primer_bloque = False

    return enviados

# =========================================================
# Transmisión principal
# =========================================================
print("[INFO] Iniciando transmisión...")

try:
    for k in range(repeticiones):
        print(f"[INFO] Repetición {k+1}/{repeticiones}")
        enviados = transmitir_burst(tx_stream, tx_md, iq_tx, bloque=bloque_envio)
        print(f"[INFO] Muestras enviadas: {enviados:,}")

        if k < repeticiones - 1:
            time.sleep(0.1)

    # Cierre formal del burst
    tx_md.start_of_burst = False
    tx_md.end_of_burst = True
    tx_md.has_time_spec = False
    tx_stream.send(np.zeros(0, dtype=np.complex64), tx_md)

    print("[INFO] Transmisión finalizada.")

except KeyboardInterrupt:
    print("\n[INFO] Transmisión interrumpida por el usuario.")

finally:
    print("[INFO] Cerrando.")