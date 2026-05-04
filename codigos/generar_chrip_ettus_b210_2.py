#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import uhd
import time
import matplotlib.pyplot as plt
from scipy import signal

# =========================================================
# Parámetros principales
# =========================================================
fc = 5.81e9            # Frecuencia central RF [Hz]
bw = 30e6              # Ancho de banda del chirp [Hz]
fs = 56e6              # Tasa de muestreo [Hz]
tx_gain = 700           # Ganancia TX
chirp_time = 1000e-6    # Duración del chirp [s]
silence_time = chirp_time  # Duración del silencio [s]
num_cycles = 500000      # Número de veces que se repetirá (chirp + ceros)
amplitude = 0.9        # Amplitud del chirp (recomendado < 1)
channel = 0            # Canal TX
dibujar = True         # Mostrar gráficas al final


# =========================================================
# Generación de chirp lineal complejo
# =========================================================
def generar_chirp_lineal_complejo(bw_hz, T, fs_hz, amplitud=0.5):
    """
    Genera un chirp lineal complejo en banda base que barre de -bw/2 a +bw/2.

    Retorna:
        chirp_bb : np.ndarray complex64
        t        : vector de tiempo del chirp [s]
        f_inst   : frecuencia instantánea teórica [Hz]
    """
    N = int(np.round(T * fs_hz))
    if N <= 0:
        raise ValueError("El número de muestras del chirp debe ser mayor que cero.")

    t = np.arange(N) / fs_hz

    f0 = -bw_hz / 2.0
    f1 =  bw_hz / 2.0
    k = (f1 - f0) / T  # pendiente [Hz/s]

    # Fase instantánea
    phi = 2.0 * np.pi * (f0 * t + 0.5 * k * t**2)

    chirp_bb = amplitud * np.exp(1j * phi)
    f_inst = f0 + k * t

    return chirp_bb.astype(np.complex64), t, f_inst


# =========================================================
# Bloque chirp + silencio
# =========================================================
def construir_bloque_tx(chirp_bb, fs_hz, silence_T):
    """
    Construye un bloque [chirp][ceros].
    """
    Nsil = int(np.round(silence_T * fs_hz))
    if Nsil < 0:
        raise ValueError("La duración del silencio no puede ser negativa.")

    silencio = np.zeros(Nsil, dtype=np.complex64)
    bloque = np.concatenate((chirp_bb, silencio)).astype(np.complex64)
    return bloque, silencio


# =========================================================
# Configuración del USRP
# =========================================================
def configurar_usrp(fc_hz, fs_hz, bw_hz, gain_db, chan=0):
    """
    Configura el USRP para TX.
    """
    usrp = uhd.usrp.MultiUSRP()

    usrp.set_tx_rate(fs_hz, chan)
    usrp.set_tx_freq(uhd.libpyuhd.types.tune_request(fc_hz), chan)
    usrp.set_tx_gain(gain_db, chan)
    usrp.set_tx_bandwidth(bw_hz, chan)

    time.sleep(0.2)

    print("=== Parámetros configurados ===")
    print(f"TX rate       : {usrp.get_tx_rate(chan):.3f} S/s")
    print(f"TX freq       : {usrp.get_tx_freq(chan):.3f} Hz")
    print(f"TX gain       : {usrp.get_tx_gain(chan):.3f} dB")
    print(f"TX bandwidth  : {usrp.get_tx_bandwidth(chan):.3f} Hz")

    st_args = uhd.usrp.StreamArgs("fc32", "sc16")
    st_args.channels = [chan]
    tx_streamer = usrp.get_tx_stream(st_args)

    return usrp, tx_streamer


# =========================================================
# Envío por streamer, sin crear un arreglo gigante
# =========================================================
def transmitir_bloque_repetido(tx_streamer, bloque_tx, ciclos):
    """
    Envía repetidamente el mismo bloque [chirp + ceros] sin usar np.tile gigante.
    """
    md = uhd.types.TXMetadata()
    md.start_of_burst = True
    md.end_of_burst = False
    md.has_time_spec = False

    max_samps = tx_streamer.get_max_num_samps()
    total_bloque = len(bloque_tx)

    print(f"Máximo por envío del streamer: {max_samps} muestras")
    print(f"Tamaño del bloque TX         : {total_bloque} muestras")
    print(f"Ciclos chirp+silencio        : {ciclos}")

    while (True):
        idx = 0
        while idx < total_bloque:
            chunk = bloque_tx[idx: idx + max_samps]
            samps_enviadas = tx_streamer.send(chunk, md)

            if samps_enviadas < 0:
                raise RuntimeError(f"Error en TX send(): código {samps_enviadas}")

            # if samps_enviadas != len(chunk):
            #     print(f"Aviso: se enviaron {samps_enviadas} de {len(chunk)} muestras en el ciclo {ciclo}")

            md.start_of_burst = False
            idx += len(chunk)

    # Final de ráfaga
    md.end_of_burst = True
    tx_streamer.send(np.zeros((1,), dtype=np.complex64), md)


# =========================================================
# Gráficas
# =========================================================
def graficar_resultados(chirp_bb, t_chirp, f_inst, bloque_tx, fs_hz):
    """
    Grafica:
    1) parte real del chirp
    2) frecuencia instantánea teórica
    3) espectrograma del bloque chirp + silencio
    """
    # ---- Gráfica temporal del chirp ----
    plt.figure(figsize=(10, 4))
    plt.plot(t_chirp * 1e6, chirp_bb.real)
    plt.xlabel("Tiempo [us]")
    plt.ylabel("Amplitud")
    plt.title("Parte real del chirp")
    plt.grid(True)
    plt.tight_layout()

    # ---- Frecuencia instantánea teórica ----
    plt.figure(figsize=(10, 4))
    plt.plot(t_chirp * 1e6, f_inst / 1e6)
    plt.xlabel("Tiempo [us]")
    plt.ylabel("Frecuencia instantánea [MHz]")
    plt.title("Frecuencia instantánea teórica del chirp")
    plt.grid(True)
    plt.tight_layout()

    # ---- Espectrograma del bloque chirp + silencio ----
    # Para señal compleja: return_onesided=False
    nperseg = min(1024, len(bloque_tx))
    noverlap = int(0.75 * nperseg)

    f, t, Sxx = signal.spectrogram(
        bloque_tx,
        fs=fs_hz,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False,
        return_onesided=False,
        scaling="density",
        mode="magnitude"
    )

    # Reordenar para que la frecuencia vaya de negativa a positiva
    f_shift = np.fft.fftshift(f)
    Sxx_shift = np.fft.fftshift(Sxx, axes=0)

    plt.figure(figsize=(11, 5))
    plt.pcolormesh(
        t * 1e6,
        f_shift / 1e6,
        20 * np.log10(Sxx_shift + 1e-12),
        shading="auto"
    )
    plt.xlabel("Tiempo [us]")
    plt.ylabel("Frecuencia [MHz]")
    plt.title("Espectrograma del bloque: chirp + silencio")
    plt.colorbar(label="Magnitud [dB]")
    plt.tight_layout()

    plt.show()


# =========================================================
# Programa principal
# =========================================================
if __name__ == "__main__":
    # 1) Generar chirp
    chirp_bb, t_chirp, f_inst = generar_chirp_lineal_complejo(
        bw_hz=bw,
        T=chirp_time,
        fs_hz=fs,
        amplitud=amplitude
    )

    # 2) Construir [chirp][ceros]
    bloque_tx, silencio = construir_bloque_tx(
        chirp_bb=chirp_bb,
        fs_hz=fs,
        silence_T=silence_time
    )

    print("=== Parámetros del bloque ===")
    print(f"Frecuencia central RF     : {fc/1e9:.6f} GHz")
    print(f"Ancho de banda chirp      : {bw/1e6:.3f} MHz")
    print(f"Frecuencia de muestreo    : {fs/1e6:.3f} MS/s")
    print(f"Duración chirp            : {chirp_time*1e6:.3f} us")
    print(f"Duración silencio         : {silence_time*1e6:.3f} us")
    print(f"Muestras del chirp        : {len(chirp_bb)}")
    print(f"Muestras de silencio      : {len(silencio)}")
    print(f"Muestras por bloque TX    : {len(bloque_tx)}")
    print(f"Amplitud                  : {amplitude}")
    print()

    # 3) Configurar USRP
    usrp, tx_streamer = configurar_usrp(
        fc_hz=fc,
        fs_hz=fs,
        bw_hz=bw,
        gain_db=tx_gain,
        chan=channel
    )

    # 4) Transmitir repetidamente el bloque sin crear un arreglo gigante
    print("\nIniciando transmisión...")
    try:
        transmitir_bloque_repetido(
            tx_streamer=tx_streamer,
            bloque_tx=bloque_tx,
            ciclos=num_cycles
        )
        print("Transmisión terminada correctamente.")
    except KeyboardInterrupt:
        print("Transmisión interrumpida por el usuario.")

    # 5) Mostrar gráficas
    if dibujar:
        graficar_resultados(
            chirp_bb=chirp_bb,
            t_chirp=t_chirp,
            f_inst=f_inst,
            bloque_tx=bloque_tx,
            fs_hz=fs
        )