"""
Created on Thu Aug 26 11:53:26 2021
Fun_MB_SS2
@author: Yanqueleth
"""

import math
import numpy as np
import pywt
import matplotlib.pyplot as plt
from pywt import wavedec
from sklearn.cluster import KMeans
import statistics
import warnings
#from CS_con_SampEn import sampen


#funcion reescalar. utilizada para los coeficientes de appx.
#los resultados de esta fucnion coinciden perfectamente con los obtenidos en la simulacion de matlab
def reescale(coef_ap_n):
    coef_ap_n=coef_ap_n-min(coef_ap_n)
    coef_ap_n=coef_ap_n/max(coef_ap_n)
    return(coef_ap_n)

#fucnion wrcof, utilizada para la señal reconstruccion de los coeff de appx.
#los resultados de esta fucnion coinciden perfectamente con los obtenidos en la simulacion de matlab
def wrcoef(X, coef_type, coeffs, wavename, level):
    N = np.array(X).size
    a, ds = coeffs[0], list(reversed(coeffs[1:]))

    if coef_type =='a':
        return pywt.upcoef('a', a, wavename, level=level)[:N]
    elif coef_type == 'd':
        return pywt.upcoef('d', ds[level-1], wavename, level=level)[:N]
    else:
        raise ValueError("Invalid coefficient type: {}".format(coef_type))
        
"""Higuchi Fractal Dimension according to:
T. Higuchi, Approach to an Irregular Time Series on the
Basis of the Fractal Theory, Physica D, 1988; 31: 277-283."""
#funcion dimension fractal de Higuchia una serie , y con un factor de descompocicion para esa señal 
#los resultados de esta funcion coinciden con los obtenidos en matlab
def HFD (serie,Kmax):

    try:
        N = len(serie)
        
        X = np.empty([N,Kmax,Kmax])
        X[:]=np.nan
        
        for k in range (1,Kmax+1):
            for m in range (1,k+1):
                limit = math.floor((N-m)/k)
                j = 1
                for i in  range(m,(m + (limit*k))+1,k):
                    X[j-1,k-1,m-1,] = serie[i-1]
                    j = j + 1
        
        L = np.zeros(Kmax)
        
        for k in range(1,Kmax+1):
            Lm = np.zeros(k,dtype=float)
            for m in range(1,k+1):
                if (math.floor((N - m)/k))==0:
                    R=math.inf
                else:
                        R = (N - 1)/(math.floor((N - m)/k) * k) 
                r1=list(X[:,k-1,m-1])
                aux=np.nan_to_num(r1)
                aux=aux[np.where(aux != 0)]
                for i in range (1,len(aux)):
                    Lm[m-1] = Lm[m-1] + abs(aux[i] - aux[i-1] )
                 
                if np.isnan(R) or np.isnan(Lm[m - 1]) or np.isnan(k) or np.isinf(R) or np.isinf(Lm[m - 1]) or np.isinf(k):
                    # Si encontramos NaN o Inf, asignamos NaN o un valor predeterminado
                    Lm[m - 1] = np.nan  
                    HFD=2
                    return HFD
                else:
                    Lm[m - 1] = (R * Lm[m - 1]) / k
              
                
            L[k-1] = sum (Lm)/k
        
        X = np.ones(k,dtype=float)
        for i in range(1,Kmax+1):
            X[i-1]=i
            
        X=1/X
        LX = np.ones(len(X),dtype=float)
        for i in range(1,len(X)+1):
            LX[i-1]=math.log(X[i-1])
        
        LL=np.ones(len(L),dtype=float)
        for i in range(1,len(L)+1):
            if L[i-1]==0:
                LL[i-1]=math.inf
            else:
                LL[i-1]=math.log(L[i-1])
        
        try:
            aux=np.polyfit(LX,LL,1)
            HFD=aux[0]
            return (HFD)
        except :
            HFD=np.nan
            return (HFD)
    except (RuntimeError, TypeError, NameError, RuntimeWarning):
        HFD=2
        return (HFD)

def principal(a,nivelwave,umbral_escalar,Kmax,umbral_total):
    
    (coef_ap_n,cons,n_clusters)=AMR(a,nivelwave,umbral_escalar)
    (idx,auxli,limites,n_clusters)=bordes(coef_ap_n,cons,n_clusters,nivelwave)
    (DFH, ocupacion, cons,auxli,ocupacion_ventanas)=MB_SS(idx,auxli,limites,n_clusters,a,cons,umbral_total,Kmax)
    return (DFH, ocupacion, cons,auxli,ocupacion_ventanas)

def principal2 (a,nivelwave,umbral_escalar,Kmax,umbral_total):
    
    (coef_ap_n,cons,n_clusters)=AMR(a,nivelwave,umbral_escalar)
    (idx,auxli,limites,n_clusters)=bordes(coef_ap_n,cons,n_clusters,nivelwave)
    (sampen2, ocupacion, cons,auxli,ocupacion_ventanas)=MB_SS2(idx,auxli,limites,n_clusters,a,cons,umbral_total,Kmax)
    return (sampen2, ocupacion, cons,auxli,ocupacion_ventanas)

#analisis multiresolucio dando como resultadolos coef app normalizados, la reconstruccion de la seña y el numero de clusters
def AMR (a,nivelwave,umbral_escalar):
    
    coeffs = wavedec(a, 'db1', level=nivelwave)
    #se hace la rescostruccion de los CA para cada frame
    cons = wrcoef(a, 'a', coeffs, 'db1', nivelwave)
    #ploteamos cada recostruccion
#    plt.show()
#    plt.plot(cons)
#    plt.xlabel('Frecuency [GHz]')
#    plt.ylabel('Power [dBm]')
#    plt.title('Señal reconstruida de los Coef. de appx.')
#    plt.show()
    #print ('la DFH para esta serie es: ',HFD)
    #Normalizamos los coeficientes de app
    coef_ap_n=abs(coeffs[0])
    ma=max(coef_ap_n)
    coef_ap_n=coef_ap_n/ma
    #buscamos si en la reconstruccion de la señal existen transmisiones
    x=0
    pos=[]
    for i in range(1,len(cons)+1):
        if cons[i-1]>umbral_escalar:
            x=x+1
            pos.append(i-1)
    
    # verificamos si los coeficietes deben reescalarse
    if len(pos)>8:
        coef_ap_n=reescale(coef_ap_n)
        n_clusters = 2
        
    else:
        n_clusters = 1

    return (coef_ap_n,cons,n_clusters)

#localizamos los Bordes de frecuencia 
def bordes(coef_ap_n,cons,n_clusters,nivelwave):
    aux = np.ones((len(coef_ap_n),2),dtype=float)
    aux[:,1]=coef_ap_n
    
    aux2=np.ones((len(coef_ap_n),2),dtype=float)
    aux2[:,0]=coef_ap_n
    for i in range(1,len(coef_ap_n)+1):
        aux2[i-1,1]=i-1
        
    kmeans = KMeans(n_clusters, n_init=10)
    
    indi=(np.where(np.isnan(aux[:,1])))
    indices=np.ones(len(indi[0]), dtype=int)
    indices[:]=indi[0]
    if len(indices)>0 and len(indices)<len(coef_ap_n): 
        aux[indices[0],1]=aux[indices[0]+1,1]
    
    indi=(np.where(np.isinf(aux[:,1])))
    indices=np.ones(len(indi[0]), dtype=int)
    indices[:]=indi[0]
    if len(indices)>0 and len(indices)<len(coef_ap_n): 
        aux[indices[0],1]=aux[indices[0]+1,1]
    
    idx = kmeans.fit_predict(aux)
    
#    plt.scatter(aux2[:,1], aux2[:,0])
#    plt.scatter(aux[:,0], aux[:,1])
#    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')    
#    plt.xlabel('Muestras')
#    plt.ylabel('Magnitud')
#    plt.title('Coeficientes de aproximación')
#    plt.show()
    
    if  n_clusters==2 and (kmeans.cluster_centers_[0,1]> kmeans.cluster_centers_[1, 1]):
        get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]
        unos=get_indexes(0,idx)
        ceros=get_indexes(1,idx)
        idx[unos]=1
        idx[ceros]=0
 
    limites=[]
    for i in range(2,len(aux)-1):
         if  idx[i-2]==0 and idx[i-1]==1 and idx[i]==1:#idx[i-1]==0  and idx[i]==1 and idx[i+1]==1:
             limites.append(i-1)
         elif idx[i-2]==1 and idx[i-1]==0 and idx[i]==0:#idx[i-1]==1 and idx[i]==0 and idx[i+1]==0:
             limites.append(i-1)
    limites.insert(0,0)
    if limites[len(limites)-1]!=len(aux)-1:
        limites.append(0)
        auxli = np.ones(len(limites),dtype=int)
        auxli[:]=limites[:]
        auxli=auxli*2**nivelwave
        limites[len(limites)-1]=len(idx)-1
        auxli[len(auxli)-1]=len(cons)-1
    else:
        auxli = np.ones(len(limites),dtype=int)
        auxli[:]=limites[:]
        auxli=auxli*2**nivelwave
        limites[len(limites)-1]=len(idx)-1
        auxli[len(auxli)-1]=len(cons)-1
        
#        #interpoolamos el resultado de kmeans
#        xvals=np.linspace(0,len(idx)-1,len(cons))
#        aux2 = np.ones(len(idx),dtype=float)
#        yinterp=np.interp(xvals,aux2,idx)
#        for i in range(1,len(yinterp)+1):
#            yinterp[i-1]=round(yinterp[i-1])
        
    return(idx,auxli,limites,n_clusters)

#aplicamos la monitorizacion del espectro multibanda en ventanas de tamaño dinamico     
def MB_SS(idx,auxli,limites,n_clusters,a,cons,umbral_total,Kmax):
    #evaluamos cada ventana de tamaño dinamico 
    ocupacion=[]
    DFH=[]
    ocupacion_ventanas=[]
    indices=(np.where(np.isnan(cons)))
    while len (indices[0])!=0:
        cons[indices[0]]=cons[indices[0]+1]
        indices=(np.where(np.isnan(cons)))
        
    

    offset=statistics.mean(cons)
    for i in range(0,len(auxli)-1):
        aa=idx[int(limites[i]):int(limites[i+1])]
        au = np.ones(len(aa),dtype=float)
        au[:]=aa
        
        if round(statistics.mean(au))==0 and n_clusters==2:
            x1=cons
        else:
            x1=a
          
        lenn=int(auxli[i+1])-int(auxli[i])
        #evaluamos por cada ventana dinamica la DFH
        aaa=HFD(x1[int(auxli[i]):int(auxli[i+1])],Kmax)
        #si la ventana localizada es tan pequeña que el resultado de la DFH es nan
        #entonces muy probablente sea ruido impulsivo y la DFH=2
        if np.isnan(aaa):
            aaa=2
            
        #if math.isinf(aaa):
            aaa=2
                
        #calculamos la ocupacion de la señal en funcion del resultado de la DFH
        #para cada ventana de tamaño dinamico
        if aaa>umbral_total:
            aux_ocupacion=np.zeros(lenn,dtype=int)+offset
            ocupacion_ventanas.extend(np.zeros(1,dtype=int))
        if aaa<umbral_total:
            aux_ocupacion=np.ones(lenn,dtype=int)*15+offset
            ocupacion_ventanas.extend(np.ones(1,dtype=int))

        #señal ocupacion    
        ocupacion.extend(aux_ocupacion)
        #señal de la dfh para todo el espectro analizado

        aux_DFH=aaa*np.ones(1,dtype=int)
        DFH.extend(aux_DFH)
    
    ocupacion.extend(np.zeros(1,dtype=int)+offset)
    #ocupacion=ocupacion-80#+offset
    #DFH.extend(np.ones(1,dtype=int)*2)
#    plt.show() 
#    plt.plot(ocupacion)    
#    plt.xlabel('Frecuencia')
#    plt.ylabel('Ocupación [-]')
#    plt.title('Monitorización MB-SS')
#    plt.show() 
    return (DFH, ocupacion, cons,auxli,ocupacion_ventanas)



#aplicamos la monitorizacion del espectro multibanda en ventanas de tamaño dinamico     
def MB_SS2(idx,auxli,limites,n_clusters,a,cons,umbral_total,Kmax):
    #evaluamos cada ventana de tamaño dinamico 
    ocupacion=[]
    sampen2=[]
    ocupacion_ventanas=[]
    indices=(np.where(np.isnan(cons)))
    while len (indices[0])!=0:
        cons[indices[0]]=cons[indices[0]+1]
        indices=(np.where(np.isnan(cons)))
        
    

    offset=statistics.mean(cons)
    for i in range(0,len(auxli)-1):
        aa=idx[int(limites[i]):int(limites[i+1])]
        au = np.ones(len(aa),dtype=float)
        au[:]=aa
        
        if round(statistics.mean(au))==0 and n_clusters==2:
            x1=cons
        else:
            x1=a
          
        lenn=int(auxli[i+1])-int(auxli[i])
        #evaluamos por cada ventana dinamica la DFH
        aaa=sampen(x1[int(auxli[i]):int(auxli[i+1])])
        #si la ventana localizada es tan pequeña que el resultado de la DFH es nan
        #entonces muy probablente sea ruido impulsivo y la DFH=2
        if np.isnan(aaa):
            aaa=2
            
        #if math.isinf(aaa):
            aaa=2
                
        #calculamos la ocupacion de la señal en funcion del resultado de la DFH
        #para cada ventana de tamaño dinamico
        if aaa>umbral_total:
            aux_ocupacion=np.zeros(lenn,dtype=int)+offset
            ocupacion_ventanas.extend(np.zeros(1,dtype=int))

        if aaa<umbral_total:
            aux_ocupacion=np.ones(lenn,dtype=int)*10+offset
            ocupacion_ventanas.extend(np.ones(1,dtype=int))

        #señal ocupacion    
        ocupacion.extend(aux_ocupacion)
        #señal de la dfh para todo el espectro analizado

        aux_DFH=aaa*np.ones(1,dtype=int)
        sampen2.extend(aux_DFH)
    
    ocupacion.extend(np.zeros(1,dtype=int)+offset)
    #ocupacion=ocupacion-80#+offset
    #DFH.extend(np.ones(1,dtype=int)*2)
#    plt.show() 
#    plt.plot(ocupacion)    
#    plt.xlabel('Frecuencia')
#    plt.ylabel('Ocupación [-]')
#    plt.title('Monitorización MB-SS')
#    plt.show() 
    return (sampen2, ocupacion, cons,auxli,ocupacion_ventanas)

""" Modulo para hacer la SampEn"""
def sampen(L):
#           
#    plt.title('Signal')
#    plt.figure(figsize=(15,8))
#    plt.show()
    
    N = len(L)
    m=2
    r=0.1*np.std(L)
    B = 0.0
    A = 0.0
        
    # Split time series and save all templates of length m
    xmi = np.array([L[i : i + m] for i in range(N - m)])
    xmj = np.array([L[i : i + m] for i in range(N - m + 1)])

    # Save all matches minus the self-match, compute B
    B = np.sum([np.sum(np.abs(xmii - xmj).max(axis=1) <= r) - 1 for xmii in xmi])

    # Similar for computing A
    m += 1
    xm = np.array([L[i : i + m] for i in range(N - m + 1)])

    A = np.sum([np.sum(np.abs(xmi - xm).max(axis=1) <= r) - 1 for xmi in xm])

    # Return SampEn  
    try:    
        aux=-np.log10(A / B)
    except RuntimeWarning:
        aux=np.nan

    return aux

    
