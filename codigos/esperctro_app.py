#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 07:48:27 2025

@author: ymt
"""

import os, sys, socket, time, pickle, pywt, Fun_MB_SS2 as MBSS, SoapySDR
import pyqtgraph as pg, numpy as np, rtlsdr as rtl, matplotlib.pyplot as plt

from threading import Thread
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSignal, QThread, QCoreApplication
from PyQt5.QtWidgets import QMessageBox

class Hilo(QThread):
    """Hilo secundario que se comunica con el graficador que captura los datos de ocupacion
    de los dispositivos SDR. Espera por la señal de STOP del servidor y termina"""
    senal=pyqtSignal()
    envia=pyqtSignal()
    limpia=pyqtSignal()

    def __init__(self, socket, rem=5000):
        """Inicializa pasando el socket y el tiempo de espera del rem"""
        super().__init__()
        self.socket = socket
        self.stop = False
        self.rem=rem
        self.termina=False

    def run(self, minuto=0, intentos=60):
        self.socket.settimeout(0.1)#Timeout para recibir de 100ms

        while minuto <= intentos and not self.stop:#Mientras no pasen 60 minutos o no se presione stop
            i = 0
            print(f"\nGrabando minuto: {minuto}")
            self.termina = False

            while i < 60 and not self.stop:#Cada segundo y miesntras no pase 1 minuto o stop
                #Muestrea, grafica y append en array
                self.senal.emit()
                print("Grabando...")

                i += 1
                try:
                    do = self.socket.recv(1024).decode('utf-8')
                    print(f"Mensaje recibido: {do}")
                    self.stop = True
                except socket.timeout:
                    self.msleep(900)  # Dormir durante 1000 milisegundos (1 segundo)
                except socket.error as e:
                    print(f"ERROR: No fue posible recibir: {e}")
                    self.msleep(900)

            #Si termina el minuto o se presiona stop, envía
            self.envia.emit()
            
            #Envía y al terminar limpia los datos de SDR
            while not self.termina:#La señal de termina se activa al terminar de recibir
                print("Se sigue recibiendo...")
                self.msleep(1)
            
            print("Esperando REM...")
            self.msleep(self.rem)
            minuto += 1

        if self.stop==True:
                print("Envío completado")
                self.quit()

class Principal(QtWidgets.QMainWindow):
    '''Define variables esenciales para el programa'''
    def __init__(self):
        super().__init__()
        """caracteristicas para la MB_SS"""
        #nivel de descomposicion para la DFH
        self.Kmax=5
        self.exit=False
        # nivel de desponcision para e analisi multiresolucion 
        self.nivelwave=3
        # umbral para la deteccion de posibles transmisiones
        self.umbral_escalar_RTL=-75
        self.umbral_escalar_hack=-85
        self.umbral_escalar_lime=-85
        #umbral para la DFH y detectar una posible transimisio 
        self.umbral_total=1.65
        self.umbral_total_cons=1.6
        self.umbral_total_cons_sampen=0.39
        self.aux_ganancia=45
        """caracteristicas de la interfaz"""
        #ancho de la ventana principal
        self.ancho_principal=800
        self.alto_principal=480
        self.ancho_graficador=300
        self.alto_graficador=350
        self.ancho_botones=130
        self.alto_botones=30
        
        self.font=QtGui.QFont("Calibri", 6)
        self.graf_color=(236,240,241)
        
        #frecuencia de muestreo
        self.fs=3.2e6#2.4e6
        self.fs_hack=20e6
        self.fs_lime=30e6
        
        #delta FS para juntar los sdr
        self.deltafs=(self.fs-2.4e6)/2
        #numero de muestras a captura
        self.N_muestras=1024
        self.N_muestras_lime=1024*144
        self.N_muestras_hack=1024*144

        self.ganancia=4
        
        """ diferentes  caracteristicas del programa"""
        #numero de puntos para la fft
        self.num_fft=512
        self.num_fft_hack=512
        self.num_fft_lime=512

        self.argumentos={
                          'rtl':{'Umbral':self.umbral_escalar_RTL, 'Fs':self.fs},
                          'lime':{'Umbral':self.umbral_escalar_lime, 'Fs':self.fs_lime},
                          'hack':{'Umbral':self.umbral_escalar_hack, 'Fs':self.fs_hack}
                        }
        
        #variable para controlar los ejes automaticos en las graficas
        self.auto=False
        #objeto timmer para controlar el retardo de actualizacion de las graficas
        self.timer = pg.QtCore.QTimer()
        self.conectado=False
        from SoapySDR import Device

        """dispositivos conectados"""
        self.disp={
            'rtl': len(rtl.RtlSdr.get_device_serial_addresses()), 
            'lime': len(Device.enumerate(dict(driver='lime'))),
            'hack': len(Device.enumerate(dict(driver='hackrf')))
        }
        
        print(self.disp)

        self.bwtotal=self.disp['rtl']*(self.fs-2*self.deltafs)+self.disp['lime']*self.fs_lime+self.disp['hack']*self.fs_hack

        """Conexion con el servidor"""
        # self.ip_servidor = '10.62.70.138'  # Cambia esto con la IP del servidor
        self.ip_servidor = 'localhost'  # Cambia esto con la IP del servidor
        self.puerto_servidor = 12345  # Cambia esto con el puerto del servidor
        self.servidor = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        #Variable que indica si la ejecucion está parada
        self.parar_grafica=False
        self.stop_variable=False
        self.ocupacion=[[np.nan], [np.nan], [np.nan]]
        self.limTotal=[[np.nan], [np.nan], [np.nan]]
        self.powerTotal=[[np.nan], [np.nan], [np.nan]]
    
    def inicializa(self):
        self.ocupacion=[[np.nan], [np.nan], [np.nan]]
        self.limTotal=[[np.nan], [np.nan], [np.nan]]
        self.powerTotal=[[np.nan], [np.nan], [np.nan]]

    '''modulo encargado de la creacion de todos los objetos que componen la interfaz'''
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setWindowTitle("SDR-UAMI MULTI-BAND SPECTRUM SENSING")
        MainWindow.resize(self.ancho_principal, self.alto_principal)
        MainWindow.setStyleSheet('background-color:rgb(242, 243, 244)')
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.horizontalLayout = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout.setObjectName("horizontalLayout")        

        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout") 

        #cuadro visualizador para las graficas  
        self.graficador = pg.PlotWidget(self.centralwidget)
        self.graficador.setObjectName("Plotter")
        self.graficador.setBackground(self.graf_color)
        self.graficador.setYRange( -85 , -50 , padding = None , update = True )

        self.horizontalLayout.addWidget(self.graficador)       
        #casilla para autorango
        self.eje_y =QtWidgets.QCheckBox ("Autorange" , self.centralwidget)
        self.eje_y.setStyleSheet(f"color: rgb(0,0,0);")
        self.verticalLayout.addWidget(self.eje_y)
        self.selector=[]
               
        """sintonziador de Fc"""
        self.titulo_sintonizador = QtWidgets.QLabel("Tuner",self.centralwidget)
        self.titulo_sintonizador.setStyleSheet("font: 11pt")
        self.titulo_sintonizador.setStyleSheet(f"color: rgb(0,0,0);")
        self.verticalLayout.addWidget(self.titulo_sintonizador)
        
        #Establece el valor minimo y maximo posible de fc para los disp conectados
        #Si hay rtl, max=2.2GHz, si hay lime y no rtl maxlime=3.5GHz, solo hack maxHack=6GHz
        #Min hay lime, minlime=30MHz, no lime pero si rtl minrtl=12MHz, solo hack minHack=1MHz

        pmax = min(2.2e9-(-self.bwtotal/2 + self.disp['rtl']*(self.fs-2*self.deltafs)/2) if self.disp['rtl'] > 0 else float('inf'),
            3.5e9-(-self.bwtotal/2+self.disp['rtl']*(self.fs-2*self.deltafs)+self.disp['lime']*self.fs_lime/2) if self.disp['lime'] > 0 else float('inf'),
            6e9 if self.disp['hack'] > 0 else float('inf'))

        pmin = max(30e6+(self.bwtotal/2)-self.disp['rtl']*(self.fs-self.deltafs)-self.disp['lime']*self.fs_lime/2 if self.disp['lime'] > 0 else float('-inf'),
            12e6+self.bwtotal/2-self.disp['rtl']*(self.fs-2*self.deltafs)/2 if self.disp['rtl'] > 0 else float('-inf'),
            1e6 if self.disp['hack'] > 0 else float('-inf'))
        
        self.sintonizador=pg.SpinBox(self.centralwidget,bounds=(pmin,pmax),suffix="Hz",siPrefix=True,step=0.5e6,decimals=5)
        self.sintonizador.setDecimals=1
        self.verticalLayout.addWidget(self.sintonizador)
        self.sintonizador.setFixedSize(self.ancho_botones,self.alto_botones)
        self.sintonizador.setStyleSheet("font: 11.5 pt")
        self.sintonizador.setValue(1e9)
        self.sintonizador.setStyleSheet(f"color: rgb(0,0,0);")
        
        """eleccion de muestras para el lime"""
        self.titulo_FFT_lime = QtWidgets.QLabel("Num samples FFT",self.centralwidget) 
        self.titulo_FFT_lime.setStyleSheet("font: 11pt")
        self.verticalLayout.addWidget(self.titulo_FFT_lime)

        self.muest_FFT_lime = QtWidgets.QComboBox(self.centralwidget)
        self.verticalLayout.addWidget(self.muest_FFT_lime)
        self.muest_FFT_lime.setFixedSize(self.ancho_botones,self.alto_botones)
        self.muest_FFT_lime.setStyleSheet("font: 11.5 pt")
        self.muest_FFT_lime.setObjectName(("Samples_FFT"))
        self.muest_FFT_lime.setStyleSheet(f"color: rgb(0,0,0);")
        self.muest_FFT_lime.addItem("512")
        self.muest_FFT_lime.addItem("1024")
        self.muest_FFT_lime.addItem("2048")
        self.muest_FFT_lime.setCurrentIndex(0)

        '''menu de ganancia para las señales'''
        self.titulo_slider_g = QtWidgets.QLabel("Added gain [dBm]",self.centralwidget) 
        self.titulo_slider_g.setStyleSheet("font: 11pt")
        self.titulo_slider_g.setStyleSheet(f"color: rgb(0,0,0);")
        self.verticalLayout.addWidget(self.titulo_slider_g)
        #slider para LA ganancia
        self.slider_g=QtWidgets.QSlider(QtCore.Qt.Horizontal,self.centralwidget)
        self.slider_g.setObjectName("slider_g")
        self.slider_g.setTickPosition(QtWidgets.QSlider.TicksBothSides) 
        self.verticalLayout.addWidget(self.slider_g)
        self.slider_g.setFixedSize(self.ancho_botones,self.alto_botones)
        self.slider_g.setMinimum(1)
        self.slider_g.setMaximum(80)
        self.slider_g.setSingleStep(2)
        self.slider_g.setValue(4)
        self.horizontalLayout.addWidget(self.graficador)

        #casilla para autorango
        self.OnLime = QtWidgets.QCheckBox ("Active Lime" , self.centralwidget)
        self.OnRtl = QtWidgets.QCheckBox ("Active Rtl" , self.centralwidget)
        self.OnHack = QtWidgets.QCheckBox ("Active Hack" , self.centralwidget)
        self.OnRtl.setChecked(1 if self.disp['rtl']>0 else 0)
        self.OnRtl.setCheckable(1 if self.disp['rtl']>0 else 0)
        self.OnLime.setChecked(1 if self.disp['lime']>0 else 0)
        self.OnLime.setCheckable(1 if self.disp['lime']>0 else 0)
        self.OnHack.setChecked(1 if self.disp['hack']>0 else 0)
        self.OnHack.setCheckable(1 if self.disp['hack']>0 else 0)

        self.OnLime.setStyleSheet(f"color: rgb(0,0,0);")
        self.OnRtl.setStyleSheet(f"color: rgb(0,0,0);")
        self.OnHack.setStyleSheet(f"color: rgb(0,0,0);")
        self.verticalLayout.addWidget(self.OnLime)
        self.verticalLayout.addWidget(self.OnRtl)
        self.verticalLayout.addWidget(self.OnHack)
  
        """botones de trabajo"""
        self.boton3 = QtWidgets.QPushButton(self.centralwidget)
        self.boton3.setObjectName("boton3")
        self.verticalLayout.addWidget(self.boton3)
        self.boton3.setText("START\nPSD ocupattion")
        self.boton3.setStyleSheet(f"color: rgb(0,0,0);")
        self.boton3.setStyleSheet("font: 11 pt")
        self.boton3.setFixedSize(self.ancho_botones,self.alto_botones+14)
        self.boton3.clicked.connect(self.comienza_psd_captura)

        #boton para salir

        """inicializador de simulacion y grabacion"""
        self.sublayout_r = QtWidgets.QHBoxLayout()
        self.sublayout_r.setObjectName("subhorizontalLayout")
        self.verticalLayout.addLayout(self.sublayout_r)

        self.salir = QtWidgets.QPushButton(self.centralwidget)
        self.salir.setObjectName("EXIT")
        self.salir.setText("EXIT")
        self.salir.setStyleSheet("font: 11 pt")
        self.sublayout_r.addWidget(self.salir)
        self.salir.setFixedSize(self.ancho_botones-80, 40)

        self.conexion = QtWidgets.QPushButton(self.centralwidget)
        self.conexion.setObjectName("Conectar")
        self.conexion.setText("Conectar")
        self.conexion.setDisabled(0)
        self.conexion.setStyleSheet("font: 11 pt")
        self.sublayout_r.addWidget(self.conexion)
        self.conexion.setFixedSize(self.ancho_botones-60, 40)
        
        #configuraciones adicionales
        self.horizontalLayout.addLayout(self.verticalLayout)
        MainWindow.setCentralWidget(self.centralwidget)

        self.conecta_mod(MainWindow)
        self.agregaSDR()
        MainWindow.setCentralWidget(self.centralwidget)
        return

    '''modulo encargado de conectar a los diferentes objetos con sus corrrespondientes modulos
    para la coneccion se emplea del metodo connect'''
    def conecta_mod(self, MainWindow):     
        self.eje_y.stateChanged.connect (self.autorango)
        self.OnLime.stateChanged.connect (self.OnOffSDR)
        self.OnHack.stateChanged.connect (self.OnOffSDR)
        self.OnRtl.stateChanged.connect (self.OnOffSDR)

        self.sintonizador.sigValueChanging.connect(self.sintoniza)

        self.slider_g.valueChanged.connect(self.actualiza_ganancia)
        self.muest_FFT_lime.activated[str].connect(self.actualiza_FFT_todos)
        self.salir.clicked.connect(self.stop_cap)
        self.conexion.clicked.connect(self.aux)
        #coneccion con el modulo inicia_timmer
        self.timer.timeout.connect(self.inicia_timmer)
        return
    
    '''modulo para iniciar y configurar los dispositivos conectados'''
    def agregaSDR(self):        
        #inicializacion de la frecuecnia central de los dispositivos
        self.fc=self.sintonizador.value()

        if self.disp['rtl']==0 and self.disp['lime']==0 and self.disp['hack']==0:
            textbox=QtWidgets.QMessageBox(self.centralwidget)
            textbox.setText("No SDR connected, try again     ")
            textbox.setWindowTitle("Error message!!!")
            textbox.setStandardButtons(QtWidgets.QMessageBox.Ok)
            textbox.buttonClicked.connect(self.cerrar)
            textbox.show()
            return
        else:
            #Dispositivo Lime
            print(self.disp['lime'])
            for i in range(self.disp['lime']):
                try:   
                    self.lime = SoapySDR.Device()
                    print("Dispositivo conectado:", self.lime.getDriverKey())

                    # self.lime = SoapySDR.Device(dict(driver='lime'))
                    self.lime.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, self.fs_lime)
                    self.lime.setGain(SoapySDR.SOAPY_SDR_RX, 0, self.slider_g.value())
                    self.rxStreamlime=self.lime.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32)
                    self.lime.activateStream(self.rxStreamlime)
                    self.muestras_lime = np.array([0]*int(self.N_muestras_lime), np.complex64)
                except Exception as e:
                    print(f"Error al conectar con el dispositivo lime: {e}")

            #Iniciar HackRF

            for i in range(self.disp['hack']):
                try:    
                    self.hack = SoapySDR.Device()  # sin filtros
                    print("Dispositivo conectado:", self.hack.getDriverKey())
                    #self.hack = SoapySDR.Device(dict(driver='hackrf'))
                    self.hack.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, self.fs_hack)
                    self.hack.setGain(SoapySDR.SOAPY_SDR_RX, 0, self.slider_g.value()+self.aux_ganancia)
                    self.rxStreamhack=self.hack.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32)
                    self.hack.activateStream(self.rxStreamhack)
                    self.muestras_hack = np.array([0]*int(self.N_muestras_hack), np.complex64)
                except Exception as e:
                    print(f"Error al conectar con el dispositivo hack: {e}")

            #iniciar RTL

            for i in range(self.disp['rtl']):
                try:    
                    self.rtl = rtl.RtlSdr()
                    self.rtl.set_sample_rate(self.fs)#3.2e6
                    self.rtl.gain=self.ganancia
                    self.muestras_rtl = np.array([0]*int(self.N_muestras), np.complex64)
                except Exception as e:
                    print(f"Error al conectar con el dispositivo rtl")  
            return
        
    def data(self):
        """Modulo que grafica y guarda en el array los datos de 3 SDR para potencia, limites y ocupacion"""
        self.graficador.setLabel('left', 'Power', units='dBm')
        self.graficador.setLabel('bottom', 'Frequency', units='Hz')

        xmin=(self.fc-self.bwtotal/2+(self.disp['rtl']*self.deltafs)/2)
        xmax=(xmin+self.bwtotal-(self.disp['rtl']*self.deltafs))
        
        self.graficador.setXRange( xmin , xmax , padding = None , update = False )
        self.graficador.enableAutoRange( axis = 'xy' , enable = self.auto  ) 
        self.graficador.showGrid(x=True, y=True)
        self.graficador.clear()
        
        #se crea un LinearRegionItem para resaltar la frecuencia sintonizada
        self.region_zoom = pg.LinearRegionItem([(self.fc)-1e6,(self.fc)+1e6])
        if self.disp['hack']==0 and self.disp['lime']==0:
            self.region_zoom = pg.LinearRegionItem([(self.fc)-0.1e6,(self.fc)+0.1e6])
        self.region_zoom.setZValue(-1)
        self.graficador.addItem(self.region_zoom)

        #Obten los datos de los 3 dispositivos
        self.muestrea_lime()
        self.muestrea_rtl()
        self.muestrea_hack()
        
        #agrego una variable mas, frecuencia, que es el dominio en frecuencia donde esta localizado
        #cada dispositivo despues utilizo una variable aux para rescatar los valores de la lista freq
        #con que corresponden a los indices marcados por la variable lim.
        
        #Grafica y guarda las variables
        pxxhack, lim_hack, ocup22_hack, sprom_hack, freq_hack = self.grafica('hack')#Primero en referencia
        pxxlime, lim_lime, ocup22_lime, sprom_lime, freq_lime= self.grafica('lime',pxxhack)#Segundo en referencia
        pxxrtl, lim_rtl, ocup22_rtl, sprom_rtl, freq_rtl = self.grafica('rtl',pxxhack if self.disp['hack'] != 0 else pxxlime)

        '''NOTA: Eliminar los argumentos de la funcion al realizar por x tiempo vs una vez '''
        #utilizo las siguientes variables como auxiliares para rescatar el valor de los bordes en frecuencia.
        aux_lim_hack=freq_hack[lim_hack] if len(lim_hack)!=0 else []
        aux_lim_lime=freq_lime[lim_lime]if len(lim_lime)!=0 else []
        aux_lim_rtl=freq_rtl[lim_rtl]if len(lim_rtl)!=0 else []

        #Al inicio los array contienen un nan, agregndo posteriormente los datos
        #La ocupación contiene la longitud del vector seguido de este
        self.ocupacion[0].append(len(ocup22_hack))
        self.ocupacion[0].extend(ocup22_hack)

        #Limites unicamente contiene los datos de limites
        self.limTotal[0].extend(aux_lim_hack)

        #La potencia contiene la longitud seguida de los datos (trama)
        self.powerTotal[0].append(len(pxxhack))
        self.powerTotal[0].extend(sprom_hack)

        #Agregar un nan después de los datos por trama
        self.ocupacion[0].append(np.nan)
        self.limTotal[0].append(np.nan)
        self.powerTotal[0].append(np.nan)
        ###########################################3

        self.ocupacion[1].append(len(ocup22_lime))
        self.ocupacion[1].extend(ocup22_lime)
        self.limTotal[1].extend(aux_lim_lime)
        self.powerTotal[1].append(len(pxxlime))
        self.powerTotal[1].extend(sprom_lime)
        self.ocupacion[1].append(np.nan)
        self.limTotal[1].append(np.nan)
        self.powerTotal[1].append(np.nan)

        self.ocupacion[2].append(len(ocup22_rtl))
        self.ocupacion[2].extend(ocup22_rtl)

        self.limTotal[2].extend(aux_lim_rtl)

        self.powerTotal[2].append(len(pxxrtl))
        self.powerTotal[2].extend(sprom_rtl)
        
        self.ocupacion[2].append(np.nan)
        self.limTotal[2].append(np.nan)
        self.powerTotal[2].append(np.nan)
    
    def aux(self):
        '''metodo que se ejecuta al presionar el boton de enviar'''
        self.threadConect = Thread(target=self.conectar, args=(15,), name='Conect')
        self.servidor = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.fc_dispositivos()
        self.conexion.setDisabled(True)
        self.threadConect.start()
        self.threadConect.join()
        #En este momento se conecta
        
        if(self.conectado==True):
            self.servidor.recv(1024).decode('utf-8')
            #Señal que indica que ya Inicia recepcion

            self.hilo = Hilo(self.servidor)
            #Cada que se emita la señal grafica y guarda variables
            #Actualiza los datos de la señal graficada
            self.hilo.senal.connect(self.data)
            self.hilo.limpia.connect(self.inicializa)
            #Cuando se emita la señal (Al recibir stop), envia
            self.hilo.envia.connect(self.envia)
            self.hilo.start()
        else:
            self.conexion.setDisabled(False)
        
    def conectar(self, intentos_maximos=15):
        '''modulo que se encarga de la conexion con la entidad central'''
        process_id = os.getpid()
        print(f"Identificador de proceso para conectar: {process_id}")
        intento_actual = 0

        while intento_actual < intentos_maximos and not self.exit:
            intento_actual += 1
            try:
                self.servidor.connect((self.ip_servidor, self.puerto_servidor))
                print(f"Conectado a {self.ip_servidor}:{self.puerto_servidor}")
                self.conectado = True
                self.conexion.setText("Conectado")
                return
            except socket.error as e:
                print(f"No fue posible conectar a {self.ip_servidor}:{self.puerto_servidor}: {e}")
                time.sleep(1)

        print(f"No fue posible conectar después de {intento_actual} intentos.")
        self.conexion.setEnabled(True)
        self.conectado = False

    # def send(self, data):
    #     '''modulo que envia la lista serializada así como la longitud'''
    #     datos_serializados=pickle.dumps(data)
    #     longitud=len(datos_serializados)
    #     try:
    #         self.servidor.sendall(struct.pack('!I', longitud)+datos_serializados)
    #     except Exception as e:
    #         print(f"Error al enviar: {e}")

    def send(self, data):
        datos=pickle.dumps(data);print(len(datos))
        length_bytes = len(datos).to_bytes(4, byteorder="big")
        try:
            self.servidor.sendall(length_bytes)
            # Enviar datos
            self.servidor.sendall(datos)
        except Exception as e:
            print(f"Error al enviar: {e}")

    def envia(self, intentos_maximos=15):
        '''Modulo que envia'''
        if not self.conectado:
            print("Desconectado")
            return

        intento_actual = 0
        while intento_actual < intentos_maximos and not self.exit:
            intento_actual += 1
            try:
                print("Enviando...")
                # Convertir a lista
                lista_enviar=[self.powerTotal, self.ocupacion, self.limTotal]

                #Envia la lista, se convierte en bytes con pickle y se envia
                self.send(lista_enviar)

                print("Completado")
                print(self.powerTotal, self.ocupacion, self.limTotal)
                self.inicializa()
                self.hilo.termina=True
                return
            except socket.error as e:
                print(f"No fue posible transmitir: {e}")
                time.sleep(1)

        print(f"No fue posible enviar después de {intento_actual} intentos.")
        self.conexion.setEnabled(True)
        
    """modulo encargado de colocar la fc de cada dispositivo, aqui hay que actualizar las Fc cuando 
    se agrega un nuevo dispositivo, y/o se modifica el valor de fc"""     
    def fc_dispositivos(self):
        self.fc = self.sintonizador.value()
       
        self.fc_rtl = self.fc - self.bwtotal/ 2 + self.disp['rtl']*(self.fs-2*self.deltafs)/2
        self.fc_lime = self.fc_rtl + self.disp['rtl']*self.fs/2+self.disp['lime']*self.fs_lime / 2
        self.fc_Hack = self.fc_lime + self.disp['lime']*self.fs_lime/2+self.disp['hack']*self.fs_hack / 2

        if self.disp['rtl']>0:
            self.muestrea_rtl()
            self.rtl.set_center_freq(self.fc_rtl+self.deltafs)
        
        if self.disp['lime']>0:
            self.muestrea_lime()
            self.lime.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, self.fc_lime)

        if self.disp['hack']>0:
            self.muestrea_hack()
            self.hack.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, self.fc_Hack)
        return
 
    '''modulo encargado de detener la captura de la senal, desconectar y cerrar la aplicacion'''
    def stop_cap(self):
        self.disp['rtl']=len(rtl.RtlSdr.get_device_serial_addresses())
        SoapySDR.Device_enumerate(dict(driver='lime'))
        self.disp['hack']=len(SoapySDR.Device_enumerate(dict(driver='hackrf')))

        if self.disp['rtl']!=0:
            self.rtl.close()
        if self.disp['hack']!=0:
            self.hack.close()
        if self.disp['lime']!=0:
            self.lime.close()
        self.exit=True

        sys.exit(0)        

    '''modulo encargado de iniciar el timer para los graficos'''
    def inicia_timmer(self):
        #este modulo solo sirve de auxiliar para inciar el timer
        return  
    
    '''modulo encargado de alternar el valor de la variable de control auto, utilizada para controlar la
    activacion de el autorango de los plots'''
    def autorango(self):
        if(self.eje_y.isChecked()):
            self.auto=True
            return
        else:
            self.auto=False
            return
    
    def OnOffSDR(self):
        self.disp['rtl']=len(rtl.RtlSdr.get_device_serial_addresses()) if self.OnRtl.isChecked() else 0
        self.disp['lime'] = len(SoapySDR.Device.enumerate({'driver': 'lime'})) if self.OnLime.isChecked() else 0
        self.disp['hack'] = len(SoapySDR.Device.enumerate({'driver': 'hackrf'})) if self.OnHack.isChecked() else 0  # Cantidad de HackRF

        self.bwtotal=self.disp['rtl']*(self.fs-2*self.deltafs)+self.disp['lime']*self.fs_lime+self.disp['hack']*self.fs_hack
        
        pmax = min(2.2e9-(-self.bwtotal/2 + self.disp['rtl']*(self.fs-2*self.deltafs)/2) if self.disp['rtl'] > 0 else float('inf'),
            3.5e9-(-self.bwtotal/2+self.disp['rtl']*(self.fs-2*self.deltafs)+self.disp['lime']*self.fs_lime/2) if self.disp['lime'] > 0 else float('inf'),
            6e9 if self.disp['hack'] > 0 else float('inf'))

        pmin = max(30e6+(self.bwtotal/2)-self.disp['rtl']*(self.fs-self.deltafs)-self.disp['lime']*self.fs_lime/2 if self.disp['lime'] > 0 else float('-inf'),
            12e6+self.bwtotal/2-self.disp['rtl']*(self.fs-2*self.deltafs)/2 if self.disp['rtl'] > 0 else float('-inf'),
            1e6 if self.disp['hack'] > 0 else float('-inf'))
        
        self.sintonizador.setMaximum(pmax)
        self.sintonizador.setMinimum(pmin)
        self.actualiza_ganancia()
        self.fc_dispositivos()
        return
    
    '''modulo encargado de actualizar el valor del sintonizador del sdr'''
    def sintoniza(self):
        self.fc=self.sintonizador.value()
        self.fc_dispositivos()
        return
    
    '''modulo encargado de cerrar la aplicacion en caso de no tener conectado un dongle'''
    def cerrar(self):
        sys.exit(0)
        return
    
    '''Modulo Encargado de cerrar acciones y confirmar el cierre'''
    def closeEvent(self, event):
        confirm_close = QMessageBox.question(
            self, "Confirmar cierre", "¿Está seguro de que desea salir?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        
        if confirm_close == QMessageBox.Yes:
            self.stop_cap()
            event.accept()
        else:
            event.ignore()

    '''modulo encargado de actualizar la ganancia de las señales capturadas'''
    def actualiza_ganancia(self):
        if self.disp['hack']>0:
            self.hack.setGain(SoapySDR.SOAPY_SDR_RX, 0, self.slider_g.value()+self.aux_ganancia)
        if self.disp['rtl']>0:
            self.rtl.set_gain(self.slider_g.value())
        if self.disp['lime']>0:
            self.lime.setGain(SoapySDR.SOAPY_SDR_RX, 0, self.slider_g.value())
         
    """ modulo encargado de actualizar el numero de puntos para la FFT del RTL_SDR"""
    def actualiza_FFT(self,num_fft):
        self.num_fft=int(num_fft)
        return
    
    """ modulo encargado de actualizar el numero de puntos para la FFT del Hack"""
    def actualiza_FFT_hack(self,num_fft_hack):
        self.num_fft_hack=int(num_fft_hack)
        return

    """ modulo encargado de actualizar el numero de puntos para la FFT del lime"""
    def actualiza_FFT_lime(self,num_fft_lime):
        self.num_fft_lime=int(num_fft_lime) 
        return

    '''modulo auxiliar para actualizar la fft de todos los dispositivos'''
    def actualiza_FFT_todos(self,num_fft_lime):
        self.actualiza_FFT(num_fft_lime)
        self.actualiza_FFT_hack(num_fft_lime)
        self.actualiza_FFT_lime(num_fft_lime)
 
    """modulo encargado de comenzar la psd de la senal general capturada"""
    def comienza_psd_captura(self):
        self.sintoniza()
        #se cierra cualquier operacion de la libreria matplotlib para evitar errores
        plt.close()
        #se activa la cuadricula para el grafico
        self.graficador.showGrid(x=True, y=True)
        #cuadro con etiquetas de cada grafica
        self.etiquetas = pg.LegendItem((100,60), offset=(70,30))  # args are (size, offset)
        self.etiquetas.setParentItem(self.graficador.graphicsItem())   # Note we do NOT call plt.addItem in this case

        if self.parar_grafica == False:
            # Iniciar el timer y establecer parar_grafica a True
            self.parar_grafica = True
            #Banderas para las etiquetas
            self.timer.timeout.connect(self.psd_captura)
            self.timer.start(300)
            self.boton3.setText("STOP\nPSD ocupattion")
        else:
            # Detener el timer y establecer parar_grafica a False
            self.timer.disconnect()
            self.parar_grafica = False
            self.boton3.setText("START\nPSD ocupattion")
        return
    
    """modulo encargado de graficar la psd de la senal capturada"""
    def psd_captura(self):
        #se ponen las etiquetas de los ejes
        self.graficador.setLabel('left', 'Power', units='dBm')
        self.graficador.setLabel('bottom', 'Frequency', units='Hz')

        xmin=(self.fc-self.bwtotal/2+(self.disp['rtl']*self.deltafs)/2)
        xmax=(xmin+self.bwtotal-(self.disp['rtl']*self.deltafs))
        
        self.graficador.setXRange( xmin , xmax , padding = None , update = False )
        self.graficador.enableAutoRange( axis = 'xy' , enable = self.auto  ) 
        self.graficador.showGrid(x=True, y=True)
        self.graficador.clear()
        
        #se crea un LinearRegionItem para resaltar la frecuencia sintonizada
        self.region_zoom = pg.LinearRegionItem([(self.fc)-1e6,(self.fc)+1e6])
        if self.disp['hack']==0 and self.disp['lime']==0:
            self.region_zoom = pg.LinearRegionItem([(self.fc)-0.1e6,(self.fc)+0.1e6])
        self.region_zoom.setZValue(-1)
        self.graficador.addItem(self.region_zoom)

        #Graficar usando señal de referencia dependiendo de los disp conectados
        #Primero toma lime si es posible, si no toma hack y al ultimo si solo hay rtl
        self.muestrea_lime()
        self.muestrea_rtl()
        self.muestrea_hack()

        #Grafica lime y guarda sus muestras, si no hay disp es []
        #Se pasan como referencia para el ajuste
        pxxhack, limites2, ocupacion22, senal_promedio2, freq_hack= self.grafica('hack')#Primero en referencia
        pxxlime, limites2, ocupacion22, senal_promedio2, freq_lime= self.grafica('lime',pxxhack)#Segundo en referencia
        self.grafica('rtl',pxxhack if self.disp['hack'] != 0 else pxxlime)

    """modulo encargado de eliminar el ruido impulsivo a traves de los coeficientes
    de detalle  y de aproximacion en el primer nivel de descomposicion"""
    def no_ruido_impulsivo(self,Pxx,freq):
        Pxx_lineal=[]
        Pxx_lineal.extend(Pxx)
        
        ni_aux=1
        aux_umbral_coeffs=0.01
        aux_umbral_coeffs2=0.4

        """descomposicion de la señal en una escala lineal"""
        coeffs_lineal = pywt.wavedec(Pxx_lineal, 'db1', level=ni_aux)
        
        """se buscan los coeficinetes  que estan entre los 2 umbrales anteriores
        lo que significa que seguramente representan ruido impulsivo"""
        for j in range(1,len(coeffs_lineal)):
            
            ma=max(abs(coeffs_lineal[j]))
            aux=abs(coeffs_lineal[j])/ma
            indices=(np.where((aux>aux_umbral_coeffs) & (aux<aux_umbral_coeffs2)))
            indices=indices[0]
            if len(indices)>0:
                indices=np.delete(indices,[len(indices)-1])
            for k in range(len(indices)-2,2,-1):
                aux1=indices[k-1]
                if indices[k]==aux1+1:
                    indices=np.delete(indices,[k])     
            
            """se ponen los coeficientesde detalle en cero y los que rodean a este 
            coeficiente"""
            if len(indices)>0:
                    
                if indices[len(indices)-1]!=len(coeffs_lineal[j])-1 and indices[len(indices)-1]!=len(coeffs_lineal[j])-2 and indices[len(indices)-1]!=len(coeffs_lineal[j])-3:
                    coeffs_lineal[j][indices]=0.000000000000001
                    coeffs_lineal[j][indices-1]=0.000000000000001
                    coeffs_lineal[j][indices+1]=0.000000000000001
                    
                    """los coeficientes de aproximacion se hacen iguales a los coeficientes que estan
                    al rededor conservando una distribucion similar a la del WAGN"""            
                    coeffs_lineal[j-1][indices]=coeffs_lineal[j-1][indices+2]
                    coeffs_lineal[j-1][indices+1]=coeffs_lineal[j-1][indices+3]
                    coeffs_lineal[j-1][indices-1]=coeffs_lineal[j-1][indices+1]
   
        """modulo para hacer deteccion con CS"""
        CA=coeffs_lineal[0]
        CD=coeffs_lineal[1]
        
        """reconstruccion y comparacion de la señal original con la señal sin ruido impulsivo"""
        Pxx_lineal_rec=pywt.waverec(coeffs_lineal,'db1','symmetric')
        return (Pxx_lineal_rec,CA)
       
    def grafica(self, disp, pxx=[]):
        #inicializo freq en cero por que al correr por primera vez el codigo freq no existe entonces existe inconsistencia 
        freq=0
        colores=['r','g', 'b', 'c', 'm', 'y', 'k', 'b']
        Pxx_lineal=[]
        limites2=[]
        ocupacion22=[]
        senal_promedio2=[]

        if disp=='rtl':fc=self.fc_rtl; color=0
        elif disp=='lime':fc=self.fc_lime; color=1
        else:fc=self.fc_Hack; color=2

        for i in range (self.disp[disp]):
            """PSD calculada con Python"""
            Pxx_lineal, freq = plt.psd(
                    self.muestras_rtl if disp == 'rtl' else
                    self.muestras_lime if disp == 'lime' else
                    self.muestras_hack,NFFT=self.num_fft_lime,
                    Fs=self.argumentos[disp]['Fs'],Fc=fc)
            
            """bloque para verificar el muestreo se hagao correctamente y no inf"""
            if any(np.isinf(Pxx_lineal)) or any(np.isnan(Pxx_lineal)):
                print('\nrtlSDR device did not make a correct sensing\n')
                break
            """bloque para eliminar el ruido impulsivo a traves de los coeficientes """
            (Pxx_lineal,CA)=self.no_ruido_impulsivo(Pxx_lineal,freq)
            Pxx_lineal=10*np.log10(Pxx_lineal/(1e-3))
            
            #Modulo para recortar pxx si el dispositivo es rtl
            if disp == 'rtl':
                Pxx_lineal = Pxx_lineal[int(len(Pxx_lineal) * (12.5/100)):-int(len(Pxx_lineal) * (12.5/100))]
                freq = freq[int(len(freq) * (12.5/100)):-int(len(freq) * (12.5/100))]

            #Ajusta respecto a referencia
            Pxx_lineal=self.ajuste(Pxx_lineal,pxx)
            """ inicia la monitorizacion con la matriz de derivacion y con SampEn"""
            (sampen2, ocupacion2, cons2, limites2, ocupacion22)=MBSS.principal(Pxx_lineal,self.nivelwave,self.argumentos[disp]['Umbral'],self.Kmax,self.umbral_total)

            for h in range(len(ocupacion22)):
                aux=Pxx_lineal[int(limites2[h]):int(limites2[h+1])]
                aux=np.mean(aux)
                senal_promedio2.extend(np.ones(1,dtype=float)*aux)

            """graficador de la señal sin ruido impulsivo"""
            #Se grafica 2 veces, una vez la señal y otra la ocupacion
            self.graficador.plot(freq,Pxx_lineal,pen=colores[color+i],name="dispositivo")
            self.graficador.plot(freq,ocupacion2,pen=colores[color+i],name="dispositivo")
        return Pxx_lineal, limites2, ocupacion22, senal_promedio2, freq 
        #agregue la lista freq como parametro de regreso 
    
    '''Funcion de ajuste de señal respecto a señal de referencia''' 
    def ajuste(self,pxx,ref):
        if len(ref)==0:
            return pxx
            
        DifPotencia = (np.max(ref) - np.min(ref)) / (np.max(pxx) - np.min(pxx))
        pxx = pxx * DifPotencia
        
        med1 = np.mean(pxx)
        med2 = np.mean(ref)
        
        if med1 <= med2:
            diferencia = med1 - med2
            pxx = pxx - diferencia
        else:
            diferencia = med2 - med1
            pxx = pxx + diferencia
        return pxx
        
    def muestrea_lime(self):
        if self.disp['lime']>0:
            self.lime.readStream(self.rxStreamlime, [self.muestras_lime], len(self.muestras_lime))
    
    def muestrea_hack(self):
        if self.disp['hack']>0:
            self.hack.readStream(self.rxStreamhack, [self.muestras_hack], self.N_muestras_hack)
            self.hack.readStream(self.rxStreamhack, [self.muestras_hack], self.N_muestras_hack)
    
    def muestrea_rtl(self):
        if self.disp['rtl']>0:
            self.muestras_rtl[:] = self.rtl.read_samples(len(self.muestras_rtl))

"""inicializador del software"""
def main():
    #Para evitar problemas de ejecucion (Wayland) con el Combobox se cambia temporalmente 
    #la variable de entorno a xcb (X Window System)
    os.environ['QT_QPA_PLATFORM'] = 'xcb'
    app = QtWidgets.QApplication(sys.argv)
    main_window = Principal()
    main_window.setupUi(main_window)
    main_window.show()
    sys.exit(app.exec_())
    
if __name__=="__main__":
    main()
    
