import numpy as np

def modulador(bits, fc, mpp):
    '''Un método que simula el esquema de 
    modulación digital BPSK.

    :param bits: Vector unidimensional de bits
    :param fc: Frecuencia de la portadora en Hz
    :param mpp: Cantidad de muestras por periodo de onda portadora
    :return: Un vector con la señal modulada
    :return: Un valor con la potencia promedio [W]
    :return: La onda portadora c(t)
    :return: La onda cuadrada moduladora (información)
    '''
    bits1 = bits.reshape(-1,4)
    # 1. Parámetros de la 'señal' de información (bits)
    N = len(bits1) # Cantidad de bits

    # 2. Construyendo un periodo de la señal portadora c(t)
    Tc = 1 / fc  # periodo [s]
    t_periodo = np.linspace(0, Tc, mpp)
    portadora_I = np.cos(2*np.pi*fc*t_periodo)
    portadora_Q = np.sin(2*np.pi*fc*t_periodo)

    # 3. Inicializar la señal modulada s(t)
    t_simulacion = np.linspace(0, N*Tc, N*mpp) 
    senal_Tx = np.zeros(t_simulacion.shape)

    moduladora_b1 = np.zeros(t_simulacion.shape)  # señal de información
    moduladora_b2 = np.zeros(t_simulacion.shape)
 
    # 4. Asignar las formas de onda según los bits (16QAM)
    for i, bit in enumerate(bits1):
        if  bit[0] == 0 and bit[1] == 0:
            senal_Tx[i*mpp : (i+1)*mpp] = portadora_I * -3
            moduladora_b1[i*mpp : (i+1)*mpp] = 0
            moduladora_b2[i*mpp : (i+1)*mpp] = 0
        if  bit[0] == 0 and bit[1] == 1:
            senal_Tx[i*mpp : (i+1)*mpp] = portadora_I * -1
            moduladora_b1[i*mpp : (i+1)*mpp] = 0
            moduladora_b2[i*mpp : (i+1)*mpp] = 1
        if  bit[0] == 1 and bit[1] == 1:
            senal_Tx[i*mpp : (i+1)*mpp] = portadora_I * 1
            moduladora_b1[i*mpp : (i+1)*mpp] = 1
            moduladora_b2[i*mpp : (i+1)*mpp] = 1
        if  bit[0] == 1 and bit[1] == 0:
            senal_Tx[i*mpp : (i+1)*mpp] = portadora_I * 3
            moduladora_b1[i*mpp : (i+1)*mpp] = 1
            moduladora_b2[i*mpp : (i+1)*mpp] = 0
        if  bit[0] == 0 and bit[1] == 0:
            senal_Tx[i*mpp : (i+1)*mpp] = portadora_Q * -3
            moduladora_b1[i*mpp : (i+1)*mpp] = 1
            moduladora_b2[i*mpp : (i+1)*mpp] = 0
        if  bit[0] == 0 and bit[1] == 1:
            senal_Tx[i*mpp : (i+1)*mpp] = portadora_Q * -1
            moduladora_b1[i*mpp : (i+1)*mpp] = 1
            moduladora_b2[i*mpp : (i+1)*mpp] = 1
        if  bit[0] == 1 and bit[1] == 1:
            senal_Tx[i*mpp : (i+1)*mpp] = portadora_Q * 1
            moduladora_b1[i*mpp : (i+1)*mpp] = 0
            moduladora_b2[i*mpp : (i+1)*mpp] = 1
        if  bit[0] == 1 and bit[1] == 0:
            senal_Tx[i*mpp : (i+1)*mpp] = portadora_Q * 3
            moduladora_b1[i*mpp : (i+1)*mpp] = 0
            moduladora_b2[i*mpp : (i+1)*mpp] = 0
            

    # 5. Calcular la potencia promedio de la señal modulada
    Pm = (1 / (N*Tc)) * np.trapz(pow(senal_Tx, 2), t_simulacion)
    Portadora_total = portadora_Q + portadora_I
    
    return senal_Tx, Pm, portadora_I, portadora_Q, moduladora_b1, moduladora_b2
  

def canal_ruidoso(senal_Tx, Pm, SNR):
    '''Un bloque que simula un medio de trans-
    misión no ideal (ruidoso) empleando ruido
    AWGN. Pide por parámetro un vector con la
    señal provieniente de un modulador y un
    valor en decibelios para la relación señal
    a ruido.

    :param senal_Tx: El vector del modulador
    :param Pm: Potencia de la señal modulada
    :param SNR: Relación señal-a-ruido en dB
    :return: La señal modulada al dejar el canal
    '''
    # Potencia del ruido generado por el canal
    Pn = Pm / pow(10, SNR/10)


    # Generando ruido auditivo blanco gaussiano
    ruido = np.random.normal(0, np.sqrt(Pn), senal_Tx.shape)

    # Señal distorsionada por el canal ruidoso
    senal_Rx = senal_Tx + ruido

    return senal_Rx
   

def demodulador(senal_Rx, portadora_Q, mpp):
    '''Un método que simula un bloque demodulador
    de señales, bajo un esquema BPSK. El criterio
    de demodulación se basa en decodificación por 
    detección de energía.

    :param senal_Rx: La señal recibida del canal
    :param portadora: La onda portadora c(t)
    :param mpp: Número de muestras por periodo
    :return: Los bits de la señal demodulada
    '''
    # Cantidad de muestras en senal_Rx
    M = len(senal_Rx)

    # Cantidad de bits en transmisión
    N = int(M / mpp)


    # Vector para bits obtenidos por la demodulación
    bits_Rx = np.zeros(N)

    # Vector para la señal demodulada
    senal_demodulada = np.zeros(M)

    # Energía de un período de la portadora
    Es = np.sum(portadora_Q**2)

    # Demodulación
    for i in range(N):
        # Producto interno de dos funciones
        producto = senal_Rx[i*mpp : (i+1)*mpp] * portadora_Q
        senal_demodulada[i*mpp : (i+1)*mpp] = producto
        
        Ep = np.sum(producto)

        # Criterio de decisión por detección de energía
        if Ep > Es*0.1:
            bits_Rx[i] = 1
        else:
            bits_Rx[i] = 0

    return bits_Rx.astype(int), senal_demodulada




def bits_a_rgb(bits_Rx, dimensiones):
    '''Un blque que decodifica el los bits
    recuperados en el proceso de demodulación

    :param: Un vector de bits 1 x k 
    :param dimensiones: Tupla con dimensiones de la img.
    :return: Un array con los pixeles reconstruidos
    '''
    # Cantidad de bits
    N = len(bits_Rx)
    

    # Se reconstruyen los canales RGB
    bits = np.split(bits_Rx, N / 8)

    # Se decofican los canales:
    canales = [int(''.join(map(str, canal)), 2) for canal in bits]
    pixeles = np.reshape(canales, dimensiones)

    return pixeles.astype(np.uint8)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


import matplotlib.pyplot as plt
import time

# Parámetros
fc = 5000  # frecuencia de la portadora
mpp = 20   # muestras por periodo de la portadora
SNR = 5    # relación señal-a-ruido del canal

# Iniciar medición del tiempo de simulación
inicio = time.time()

# 1. Importar y convertir la imagen a trasmitir
imagen_Tx = fuente_info('arenal.jpg')
dimensiones = imagen_Tx.shape


# 2. Codificar los pixeles de la imagen
bits_Tx = rgb_a_bit(imagen_Tx)

# 3. Modular la cadena de bits usando el esquema BPSK
senal_Tx, Pm, portadora_I, portadora_Q, moduladora_b1, moduladora_b2 = modulador(bits_Tx, fc, mpp) 

# 4. Se transmite la señal modulada, por un canal ruidoso
senal_Rx = canal_ruidoso(senal_Tx, Pm, SNR)

# 5. Se desmodula la señal recibida del canal
bits_Rx, senal_demodulada = demodulador(senal_Rx, portadora_Q, mpp)

# 6. Se visualiza la imagen recibida 
imagen_Rx = bits_a_rgb(bits_Rx, dimensiones)

Fig = plt.figure(figsize=(10,6))

# Cálculo del tiempo de simulación
print('Duración de la simulación: ', time.time() - inicio)

# 7. Calcular número de errores
errores = sum(abs(bits_Tx - bits_Rx))
BER = errores/len(bits_Tx)
print('{} errores, para un BER de {:0.4f}.'.format(errores, BER))

# Mostrar imagen transmitida
ax = Fig.add_subplot(1, 2, 1)
imgplot = plt.imshow(imagen_Tx)
ax.set_title('Transmitido')

# Mostrar imagen recuperada
ax = Fig.add_subplot(1, 2, 2)
imgplot = plt.imshow(imagen_Rx)
ax.set_title('Recuperado')

Fig.tight_layout()

plt.imshow(imagen_Rx)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


# Visualizar el cambio entre las señales
fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, sharex=True, figsize=(14, 7))

moduladora_T= moduladora_b1 - moduladora_b2
# La onda cuadrada moduladora (bits de entrada)
ax1.plot(moduladora_T[0:600], color='r', lw=2) 
ax1.set_ylabel('$b(t)$')

# La señal modulada por BPSK
ax2.plot(senal_Tx[0:600], color='g', lw=2) 
ax2.set_ylabel('$s(t)$')

# La señal modulada al dejar el canal
ax3.plot(senal_Rx[0:600], color='b', lw=2) 
ax3.set_ylabel('$s(t) + n(t)$')

# La señal demodulada
ax4.plot(senal_demodulada[0:600], color='m', lw=2) 
ax4.set_ylabel('$b^{\prime}(t)$')
ax4.set_xlabel('$t$ / milisegundos')
fig.tight_layout()
plt.show()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
# Creación del vector de tiempo.
Tiempo = 100        
tiempo_final = 10    
frecuencia = 10000       
mpp = 100
t = np.linspace(0, tiempo_final, Tiempo) #Creación del vector de tiempo como tal.

Tc = 1 / fc  # Aquí se define el periodo.
tiempo_p = np.linspace(0, Tc, mpp) 
portadora_Q = np.sin(2*np.pi*frecuencia*tiempo_p)
portadora_I = np.cos(2*np.pi*frecuencia*tiempo_p)

# Inicialización del proceso aleatorio Señal_Tx (H(t)) con N realizaciones
N = 4
H_t = np.empty((N, len(t)))	# N funciones del tiempo h(t) con T puntos

# Creación de las muestras del proceso x(t) (A y Z independientes)
for i in range(N):
    if i == 0:
        h_t = portadora_Q + portadora_I
    if i == 1:
        h_t = portadora_Q - portadora_I
    if i == 2:
        h_t = portadora_I - portadora_Q
    if i == 3:
        h_t = (portadora_Q * -1) + (portadora_I * -1)
    H_t[i,:] = h_t
    plt.plot(t, h_t)

# Promedio de las N realizaciones en cada instante 
P = [np.mean(H_t[:,i]) for i in range(len(t))]
plt.plot(t, P, lw=6)

#  realizaciones, y su promedio calculado y teórico
plt.title('Realizaciones de SeñalTx')
plt.xlabel('$t$')
plt.ylabel('$H(t)$')
plt.show()

# T valores de desplazamiento tao
desplazamiento = np.arange(Tiempo)
taos = desplazamiento/tiempo_final

# Inicialización de matriz de valores de correlación para la función SeñalTx.
correlacion = np.empty((len(desplazamiento)))

# Nueva figura para la autocorrelación
plt.figure()

# Cálculo de correlación para cada valor de tao
for i, tao in enumerate(desplazamiento):
    correlacion[i] = np.correlate(senal_Tx, np.roll(senal_Tx, tao))/Tiempo
plt.plot(taos, correlacion)

# Gráficas de correlación para cada realización y la
plt.title('Función de autocorrelación')
plt.xlabel(r'$\tau$')
plt.ylabel(r'SeñalTx$(\tau)$')
plt.show()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Parte 4.3
from scipy import fft

# Tf de fourier
senal_f = fft(senal_Tx)

# Muestras de la señal
Numero_muestras = len(senal_Tx)

# Número de simbolos 
Numero_simbolos = Numero_muestras // mpp

# tiempo del simbolo
Tiempo_simbolo = 1 / fc

# Tiempo entre muestras
Timepo_muestras = Tiempo_simbolo / mpp

# Tiempo de simulación
T = Numero_simbolos * Tiempo_simbolo

# Espacio de frecuencias
f = np.linspace(0.0, 1.0/(2.0*Timepo_muestras), Numero_muestras//2)

#Grafico
plt.plot(f, 2.0/Numero_muestras * np.power(np.abs(senal_f[0:Numero_muestras//2]), 2))
plt.xlim(0, 20000)
plt.grid()
plt.show