#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import uhd
import time

# =========================
# Parámetros
# =========================
fc = 5.81e9
fs = 40e6          # prueba primero con 10 MS/s
bw = fs         # BW RF deseado
tx_gain = 70       # NO empieces con 70
ampl = 20.15
tx_chan = 0
antenna = "TX/RX"
chunk_size = 1000   # más razonable que 100000 para streaming continuo
args = ""            # o "type=b200"

# =========================
# Crear USRP
# =========================
usrp = uhd.usrp.MultiUSRP(args)

print("Configurando USRP B210...")
usrp.set_tx_rate(fs, tx_chan)
usrp.set_tx_freq(uhd.libpyuhd.types.tune_request(fc), tx_chan)
usrp.set_tx_gain(tx_gain, tx_chan)
usrp.set_tx_bandwidth(bw, tx_chan)
usrp.set_tx_antenna(antenna, tx_chan)

time.sleep(0.5)

print(f"TX Rate       : {usrp.get_tx_rate(tx_chan)/1e6:.3f} MS/s")
print(f"TX Freq       : {usrp.get_tx_freq(tx_chan)/1e9:.6f} GHz")
print(f"TX Gain       : {usrp.get_tx_gain(tx_chan):.2f} dB")
print(f"TX Bandwidth  : {usrp.get_tx_bandwidth(tx_chan)/1e6:.3f} MHz")
print(f"TX Antenna    : {usrp.get_tx_antenna(tx_chan)}")

# =========================
# Stream TX
# =========================
st_args = uhd.usrp.StreamArgs("fc32", "sc16")
st_args.channels = [tx_chan]
tx_streamer = usrp.get_tx_stream(st_args)

tx_md = uhd.types.TXMetadata()
tx_md.start_of_burst = True
tx_md.end_of_burst = False
tx_md.has_time_spec = False

# =========================
# Pre-generar buffers de ruido
# =========================
rng = np.random.default_rng(1234)

def gen_complex_noise(n, amplitude=0.15):
    x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
    x /= np.sqrt(2.0)
    x *= amplitude
    return x

# Reutilizamos varios buffers para quitar carga al CPU
noise_buffers = [gen_complex_noise(chunk_size, ampl) for _ in range(8)]

print("\nTransmitiendo ruido continuamente... Ctrl+C para detener.\n")

device_lost = False
idx = 0

try:
    while True:
        buff = noise_buffers[idx]
        sent = tx_streamer.send(buff, tx_md)

        if sent != len(buff):
            print(f"[WARN] Solo se enviaron {sent} de {len(buff)} muestras")

        tx_md.start_of_burst = False
        idx = (idx + 1) % len(noise_buffers)

except KeyboardInterrupt:
    print("\nDeteniendo por teclado...")

except RuntimeError as e:
    device_lost = True
    print("\n[ERROR] Falló la transmisión.")
    print(str(e))

finally:
    if not device_lost:
        try:
            tx_md.start_of_burst = False
            tx_md.end_of_burst = True
            tx_streamer.send(np.zeros(1, dtype=np.complex64), tx_md)
        except Exception as e:
            print(f"[WARN] No se pudo mandar end_of_burst: {e}")

    print("TX terminada.")