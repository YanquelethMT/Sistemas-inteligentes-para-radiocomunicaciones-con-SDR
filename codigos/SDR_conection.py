#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 11:45:06 2025

@author: ym-t
"""
from rtlsdr import RtlSdr
import time
# Inicializa el SDR
sdr = RtlSdr()
print("SDR conectado")

time.sleep(1)
# Cierra el dispositivo cuando termines
sdr.close()

# Imprime algunas muestras
print("SDR desconectado")
