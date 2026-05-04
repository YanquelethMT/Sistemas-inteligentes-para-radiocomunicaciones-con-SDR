#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 14:57:52 2026

@author: ymt
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import uhd
import time

# ==========================================
# Parámetros
# ==========================================
fc_list = [5.75e9, 5.81e9]   # frecuencias a alternar
dwell_time = 0.01             # tiempo en cada frecuencia [s]

fs = 40e6                    # empieza estable
bw = fs
tx_gain = 70
ampl = 35.12
tx_chan = 0
antenna = "TX/RX"
chunk_size = 1000
num_buffers = 1
args = ""                    # o "type=b200"

# opcional: unos ms de silencio al cambiar frecuencia
mute_time = 0.0             # 20 ms

# ==========================================
# Crear USRP
# ==========================================
usrp = uhd.usrp.MultiUSRP(args)

print("Configurando USRP B210...")
usrp.set_tx_rate(fs, tx_chan)
usrp.set_tx_gain(tx_gain, tx_chan)
usrp.set_tx_bandwidth(bw, tx_chan)
usrp.set_tx_antenna(antenna, tx_chan)
usrp.set_tx_freq(uhd.libpyuhd.types.tune_request(fc_list[0]), tx_chan)

time.sleep(0.5)

print(f"TX Rate       : {usrp.get_tx_rate(tx_chan)/1e6:.3f} MS/s")
print(f"TX Freq       : {usrp.get_tx_freq(tx_chan)/1e9:.6f} GHz")
print(f"TX Gain       : {usrp.get_tx_gain(tx_chan):.2f} dB")
print(f"TX Bandwidth  : {usrp.get_tx_bandwidth(tx_chan)/1e6:.3f} MHz")
print(f"TX Antenna    : {usrp.get_tx_antenna(tx_chan)}")

# ==========================================
# TX Stream
# ==========================================
st_args = uhd.usrp.StreamArgs("fc32", "sc16")
st_args.channels = [tx_chan]
tx_streamer = usrp.get_tx_stream(st_args)

tx_md = uhd.types.TXMetadata()
tx_md.start_of_burst = True
tx_md.end_of_burst = False
tx_md.has_time_spec = False

# ==========================================
# Generación de ruido
# ==========================================
rng = np.random.default_rng(1234)

def gen_complex_noise(n, amplitude=0.12):
    x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64)
    x /= np.sqrt(2.0)   # RMS aprox. unitaria
    x *= amplitude
    return x

noise_buffers = [gen_complex_noise(chunk_size, ampl) for _ in range(num_buffers)]
zero_buffer = np.zeros(chunk_size, dtype=np.complex64)

# ==========================================
# Loop TX con hopping
# ==========================================
print("\nTransmitiendo ruido con conmutación 5.75 <-> 5.81 GHz... Ctrl+C para detener.\n")

device_lost = False
buf_idx = 0
fc_idx = 0
current_fc = fc_list[fc_idx]

t_next_switch = time.time() + dwell_time

try:
    while True:
        now = time.time()

        # ---- cambiar de frecuencia cuando toque
        if now >= t_next_switch:
            # silenciar un instante para que el retune no quede tan brusco
            n_mute_blocks = max(1, int((mute_time * fs) / chunk_size))
            for _ in range(n_mute_blocks):
                tx_streamer.send(zero_buffer, tx_md)
                tx_md.start_of_burst = False

            fc_idx = (fc_idx + 1) % len(fc_list)
            current_fc = fc_list[fc_idx]

            print(f"Cambiando a {current_fc/1e9:.6f} GHz")
            usrp.set_tx_freq(uhd.libpyuhd.types.tune_request(current_fc), tx_chan)

            t_next_switch = now + dwell_time

        # ---- mandar ruido
        buff = noise_buffers[buf_idx]
        sent = tx_streamer.send(buff, tx_md)

        if sent != len(buff):
            print(f"[WARN] Se enviaron {sent} de {len(buff)} muestras")

        tx_md.start_of_burst = False
        buf_idx = (buf_idx + 1) % num_buffers

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