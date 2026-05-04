import SoapySDR

# Buscar todos los SDR conectados
sdrs_conectados = SoapySDR.Device.enumerate()
print(sdrs_conectados)
print(len(sdrs_conectados))
# Filtrar los que son RTL-SDR

print("🔍 SDRs encontrados:")
for i, sdr in enumerate(sdrs_conectados):
    print(f"[{i}] {sdr}")
