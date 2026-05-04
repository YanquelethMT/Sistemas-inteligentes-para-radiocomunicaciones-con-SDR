#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analizador de espectro rápido para Ettus USRP B210.
- UHD directo (sin Soapy para abrir el radio)
- GUI con PyQtGraph
- Cambio de frecuencia central en tiempo real
- Recepción continua; sólo hace retune cuando cambias fc

Uso típico:
    python b210_spectrum_fast.py --fc-mhz 5745 --rate 20e6 --gain 50
"""

import sys
import argparse
import threading
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets
import uhd


class SpectrumWorker(QtCore.QThread):
    spectrum_ready = QtCore.pyqtSignal(object, object, float)
    status = QtCore.pyqtSignal(str)
    error = QtCore.pyqtSignal(str)

    def __init__(
        self,
        device_args="type=b200",
        sample_rate=20e6,
        gain=50.0,
        bandwidth=None,
        antenna="TX/RX",
        fft_size=2048,
        avg_count=4,
        center_freq=2.437e9,
        channel=0,
        parent=None,
    ):
        super().__init__(parent)
        self.device_args = device_args
        self.sample_rate = float(sample_rate)
        self.gain = float(gain)
        self.bandwidth = float(bandwidth) if bandwidth is not None else None
        self.antenna = antenna
        self.fft_size = int(fft_size)
        self.avg_count = int(avg_count)
        self.center_freq = float(center_freq)
        self.channel = int(channel)

        self._lock = threading.Lock()
        self._pending_freq = None
        self._running = True

        self.window = np.hanning(self.fft_size).astype(np.float32)
        self.rel_freq_mhz = (
            np.fft.fftshift(np.fft.fftfreq(self.fft_size, d=1.0 / self.sample_rate)) / 1e6
        )

        self.frame = np.empty(self.fft_size, dtype=np.complex64)
        self.tmp = np.empty(max(4096, self.fft_size), dtype=np.complex64)

        self.usrp = None
        self.rx_stream = None
        self.metadata = None

    def request_tune(self, freq_hz: float):
        with self._lock:
            self._pending_freq = float(freq_hz)

    def stop(self):
        with self._lock:
            self._running = False

    def _is_running(self):
        with self._lock:
            return self._running

    def _read_exact(self, out: np.ndarray, timeout_s: float = 0.15) -> bool:
        filled = 0
        n_total = out.size

        while filled < n_total and self._is_running():
            got = self.rx_stream.recv(self.tmp, self.metadata, timeout_s)
            if got is None:
                got = 0

            if got > 0:
                take = min(got, n_total - filled)
                out[filled:filled + take] = self.tmp[:take]
                filled += take
            else:
                continue

        return filled == n_total

    def _retune(self, freq_hz: float):
        self.usrp.set_rx_freq(uhd.libpyuhd.types.tune_request(freq_hz), self.channel)
        self.center_freq = float(self.usrp.get_rx_freq(self.channel))
        self.status.emit(f"fc = {self.center_freq/1e6:.3f} MHz")

        for _ in range(2):
            _ = self.rx_stream.recv(self.tmp, self.metadata, 0.03)

    def run(self):
        try:
            self.status.emit(f"Abriendo USRP con args: {self.device_args}")
            self.usrp = uhd.usrp.MultiUSRP(self.device_args)

            self.usrp.set_rx_rate(self.sample_rate, self.channel)
            self.usrp.set_rx_gain(self.gain, self.channel)

            if self.bandwidth is not None:
                try:
                    self.usrp.set_rx_bandwidth(self.bandwidth, self.channel)
                except Exception as e:
                    self.status.emit(f"[WARN] BW no aplicado: {e}")

            if self.antenna:
                try:
                    self.usrp.set_rx_antenna(self.antenna, self.channel)
                except Exception as e:
                    self.status.emit(f"[WARN] Antena no aplicada: {e}")

            st_args = uhd.usrp.StreamArgs("fc32", "sc16")
            st_args.channels = [self.channel]
            self.rx_stream = self.usrp.get_rx_stream(st_args)
            self.metadata = uhd.types.RXMetadata()

            cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
            cmd.stream_now = True
            self.rx_stream.issue_stream_cmd(cmd)

            self._retune(self.center_freq)

            psd_acc = np.empty(self.fft_size, dtype=np.float64)

            while self._is_running():
                with self._lock:
                    pending = self._pending_freq
                    self._pending_freq = None

                if pending is not None:
                    self._retune(pending)

                psd_acc.fill(0.0)
                n_frames = 0

                for _ in range(self.avg_count):
                    if not self._read_exact(self.frame):
                        break

                    x = self.frame * self.window
                    X = np.fft.fftshift(np.fft.fft(x, self.fft_size))
                    psd_acc += (X.real * X.real + X.imag * X.imag)
                    n_frames += 1

                if n_frames == 0:
                    continue

                psd_db = 10.0 * np.log10(psd_acc / n_frames + 1e-12)
                freq_mhz = self.rel_freq_mhz + (self.center_freq / 1e6)

                self.spectrum_ready.emit(freq_mhz, psd_db, self.center_freq)

        except Exception as e:
            self.error.emit(str(e))

        finally:
            try:
                if self.rx_stream is not None:
                    stop_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
                    self.rx_stream.issue_stream_cmd(stop_cmd)
            except Exception:
                pass


class SpectrumWindow(QtWidgets.QMainWindow):
    def __init__(
        self,
        device_args="type=b200",
        fc_mhz=2437.0,
        sample_rate=20e6,
        gain=50.0,
        bandwidth=None,
        antenna="TX/RX",
        fft_size=2048,
        avg_count=4,
    ):
        super().__init__()
        self.setWindowTitle("USRP B210 - Espectro en tiempo real")
        self.resize(1200, 700)

        pg.setConfigOptions(antialias=False)

        self.worker = SpectrumWorker(
            device_args=device_args,
            sample_rate=sample_rate,
            gain=gain,
            bandwidth=bandwidth,
            antenna=antenna,
            fft_size=fft_size,
            avg_count=avg_count,
            center_freq=fc_mhz * 1e6,
            channel=0,
        )

        self._build_ui(fc_mhz, sample_rate, gain, fft_size, avg_count)
        self._connect_signals()
        self.worker.start()

    def _build_ui(self, fc_mhz, sample_rate, gain, fft_size, avg_count):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)

        root = QtWidgets.QVBoxLayout(central)
        controls = QtWidgets.QHBoxLayout()

        self.fc_edit = QtWidgets.QLineEdit(f"{fc_mhz:.3f}")
        self.fc_edit.setMaximumWidth(120)
        self.fc_edit.setToolTip("Frecuencia central en MHz")

        self.apply_btn = QtWidgets.QPushButton("Aplicar fc")
        self.minus_btn = QtWidgets.QPushButton("−")
        self.plus_btn = QtWidgets.QPushButton("+")

        self.step_combo = QtWidgets.QComboBox()
        for val in ["0.1", "0.5", "1", "2", "5", "10", "20"]:
            self.step_combo.addItem(val)
        self.step_combo.setCurrentText("5")

        self.actual_fc_lbl = QtWidgets.QLabel("fc actual: --")
        self.info_lbl = QtWidgets.QLabel(
            f"Fs={sample_rate/1e6:.1f} MS/s | Gain={gain:.1f} dB | FFT={fft_size} | Avg={avg_count}"
        )

        controls.addWidget(QtWidgets.QLabel("fc [MHz]:"))
        controls.addWidget(self.fc_edit)
        controls.addWidget(self.apply_btn)
        controls.addSpacing(10)
        controls.addWidget(self.minus_btn)
        controls.addWidget(self.plus_btn)
        controls.addWidget(QtWidgets.QLabel("Paso [MHz]:"))
        controls.addWidget(self.step_combo)
        controls.addStretch(1)
        controls.addWidget(self.info_lbl)
        controls.addSpacing(20)
        controls.addWidget(self.actual_fc_lbl)

        root.addLayout(controls)

        self.plot = pg.PlotWidget()
        self.plot.setBackground("k")
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setLabel("bottom", "Frecuencia", units="MHz")
        self.plot.setLabel("left", "Potencia", units="dB")
        self.plot.setYRange(-120, 10)
        self.curve = self.plot.plot(pen=pg.mkPen((0, 255, 0), width=1))
        root.addWidget(self.plot)

        self.statusBar().showMessage("Iniciando...")

    def _connect_signals(self):
        self.apply_btn.clicked.connect(self.apply_frequency)
        self.fc_edit.returnPressed.connect(self.apply_frequency)
        self.minus_btn.clicked.connect(lambda: self.step_frequency(-1))
        self.plus_btn.clicked.connect(lambda: self.step_frequency(+1))

        self.worker.spectrum_ready.connect(self.on_spectrum)
        self.worker.status.connect(self.on_status)
        self.worker.error.connect(self.on_error)

    def _current_step_mhz(self) -> float:
        return float(self.step_combo.currentText())

    def apply_frequency(self):
        try:
            fc_mhz = float(self.fc_edit.text().strip())
        except ValueError:
            self.statusBar().showMessage("fc inválida")
            return

        self.worker.request_tune(fc_mhz * 1e6)
        self.statusBar().showMessage(f"Retune solicitado a {fc_mhz:.3f} MHz")

    def step_frequency(self, sign: int):
        try:
            fc_mhz = float(self.fc_edit.text().strip())
        except ValueError:
            fc_mhz = 0.0

        fc_mhz += sign * self._current_step_mhz()
        self.fc_edit.setText(f"{fc_mhz:.3f}")
        self.apply_frequency()

    def on_spectrum(self, freq_mhz, psd_db, center_freq_hz):
        self.curve.setData(freq_mhz, psd_db)
        self.plot.setXRange(freq_mhz[0], freq_mhz[-1], padding=0.0)
        self.actual_fc_lbl.setText(f"fc actual: {center_freq_hz/1e6:.3f} MHz")

    def on_status(self, msg: str):
        self.statusBar().showMessage(msg)

    def on_error(self, msg: str):
        QtWidgets.QMessageBox.critical(self, "Error", msg)
        self.statusBar().showMessage(f"Error: {msg}")

    def keyPressEvent(self, event):
        if event.key() == QtCore.Qt.Key_Left:
            self.step_frequency(-1)
            event.accept()
            return
        if event.key() == QtCore.Qt.Key_Right:
            self.step_frequency(+1)
            event.accept()
            return
        if event.key() == QtCore.Qt.Key_PageDown:
            self.fc_edit.setText(f"{float(self.fc_edit.text()) - 10*self._current_step_mhz():.3f}")
            self.apply_frequency()
            event.accept()
            return
        if event.key() == QtCore.Qt.Key_PageUp:
            self.fc_edit.setText(f"{float(self.fc_edit.text()) + 10*self._current_step_mhz():.3f}")
            self.apply_frequency()
            event.accept()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event):
        self.worker.stop()
        self.worker.wait(2000)
        event.accept()


def parse_args():
    p = argparse.ArgumentParser(description="Espectro rápido para Ettus USRP B210")
    p.add_argument("--args", default="type=b200", help='Args UHD. Ejemplo: "type=b200,serial=34D1292"')
    p.add_argument("--fc-mhz", type=float, default=2437.0, help="Frecuencia central inicial en MHz")
    p.add_argument("--rate", type=float, default=20e6, help="Sample rate en Hz")
    p.add_argument("--gain", type=float, default=50.0, help="Ganancia RX en dB")
    p.add_argument("--bw", type=float, default=None, help="Bandwidth RX en Hz")
    p.add_argument("--antenna", default="TX/RX", help='Antena RX, por ejemplo "TX/RX" o "RX2"')
    p.add_argument("--fft", type=int, default=2048, help="Tamaño FFT")
    p.add_argument("--avg", type=int, default=4, help="Promedios por actualización")
    return p.parse_args()


def main():
    args = parse_args()

    app = QtWidgets.QApplication(sys.argv)
    win = SpectrumWindow(
        device_args=args.args,
        fc_mhz=args.fc_mhz,
        sample_rate=args.rate,
        gain=args.gain,
        bandwidth=args.bw,
        antenna=args.antenna,
        fft_size=args.fft,
        avg_count=args.avg,
    )
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
