#import tensorflow as tf
#import tensorflow_datasets as tfds
#import keras
#import glob
#from keras import layers, models
#from keras.models import Sequential
#from keras.layers import Dense,Conv2D, Dropout,Activation,MaxPooling2D,Flatten
#from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import load_model
#import pathlib
#import datetime
import os
from turtle import pd
#import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
np.set_printoptions(precision=4)
import os
from termcolor import colored



#Preparando los datos para meterlos a la red y probarla.

#Ya tengo los csv listos, los importaré
#Controles
os.chdir(r'C:\Users\jairm\Downloads\Controles\Controles\Estudios_nuevos\SE34_PTDAH_OC')
filename = 'Controles_nuevos.csv'
Base_R1 = pd.read_csv(filename, header=None) #De una vez le di el nombre que necesitaba.


#TDAH
os.chdir(r'C:\Users\jairm\Downloads\TDAH\TDAH\Nuevos_estudios\SE29_PTDAH_OC')
filename = 'TDAH_nuevos.csv'
Base_R2 = pd.read_csv(filename, header=None)

#Transformada de Fourier
import scipy.fftpack as fourier 

#print(f'Es el primer type {type(Base_R1)}')
Base_R1=fourier.fft(Base_R1) #se aplica la transformada de fourier a la base1 
#de controles 
#print(f'Es el segundo type {type(Base_R1)}')
Base_R1=abs(Base_R1) #se calcula es valor absoluto de la transformada de fourier
#de la base1 

Base_R1=pd.DataFrame(Base_R1) #volvemos a generar un dataframe del valor absoluto
#de la base1. Esto se corrigio del codigo original 
#display(Base_R1)
"""Acá este dataframe se ve bien. Después lo vuelve a pasar a df, pero se empieza a ver raro. Quizás de eso vienen los problemas."""
#print(f'Es el tercer type {type(Base_R1)}')

Base_R2=fourier.fft(Base_R2) #se aplica la transformada de fourier a la base2 
#de TDAH
Base_R2=abs(Base_R2) #se calcula es valor absoluto de la transformada de fourier
#de la base2
Base_R2 = pd.DataFrame(Base_R2) #volvemos a generar un dataframe del valor absoluto
#de la base2. Esto se corrigio del codigo original 

'''Hasta acá ya revisé y sé lo que hace el código...'''
#controles 
#se crea una lista vacia para llenar con los errores y los errores se 
#etiquetan con 'y'

#Base_R1_controles=pd.DataFrame(Base_R1) #nuevo dataframe de la baseR1 #Comenté esta línea
#La siguiente línea será la nueva... El motivo se encuentra en el comentario en verde
"""Desde acá ya se ven raros los df."""

'''print(Base_R1.isnull().values.any())
exit()'''
"""Efectivamente, acá es donde empiezan los errores. Solamente lo voy a reasignar
y veré el efecto sobre la red..."""

os.chdir(r'C:\Users\jairm\Downloads')
model = 'red_bin_2_epocas_entrenando_mas_datos_v4_60_epocas.h5'
print(colored(f'{model}', 'blue'))
model = load_model(model)


print(colored('################# DATOS ANTES VISTOS ##################', 'magenta'))
#Predicción para controles
prediction = model.predict(Base_R1)
aciertos = 0
desaciertos = 0
for i in prediction:
    if i >=0.5:
        aciertos += 1
    else:
        desaciertos += 1
print(f'Los aciertos para controles son {aciertos}, los desaciertos para controles son {desaciertos}')

#Predicción para TDAH
prediction = model.predict(Base_R2)
aciertos = 0
desaciertos = 0
for i in prediction:
    if i <=0.5:
        aciertos += 1
    else:
        desaciertos += 1
print(f'Los aciertos para TDAH son {aciertos}, los desaciertos para TDAH son {desaciertos}')

print(colored('################# DATOS NUNCA ANTES VISTOS ##################', 'magenta'))

os.chdir(r'C:\Users\jairm\Downloads\Controles_csv\SE02_PTDAH_OC')
filename = 'controles.csv'
Base_R1 = pd.read_csv(filename, header=None) #De una vez le di el nombre que necesitaba. 



#TDAH
#Acá empiezal a parte borrada 2

os.chdir(r'C:\Users\jairm\Downloads\TDAH_csv\SE05_PTDAH_OC')
filename = 'TDAH.csv'
Base_R2 = pd.read_csv(filename, header=None)

#Transformada de Fourier
import scipy.fftpack as fourier 

#print(f'Es el primer type {type(Base_R1)}')
Base_R1=fourier.fft(Base_R1) #se aplica la transformada de fourier a la base1 
#de controles 
#print(f'Es el segundo type {type(Base_R1)}')
Base_R1=abs(Base_R1) #se calcula es valor absoluto de la transformada de fourier
#de la base1 

Base_R1=pd.DataFrame(Base_R1) #volvemos a generar un dataframe del valor absoluto
#de la base1. Esto se corrigio del codigo original 
#display(Base_R1)
"""Acá este dataframe se ve bien. Después lo vuelve a pasar a df, pero se empieza a ver raro. Quizás de eso vienen los problemas."""
#print(f'Es el tercer type {type(Base_R1)}')

Base_R2=fourier.fft(Base_R2) #se aplica la transformada de fourier a la base2 
#de TDAH
Base_R2=abs(Base_R2) #se calcula es valor absoluto de la transformada de fourier
#de la base2
Base_R2 = pd.DataFrame(Base_R2) #volvemos a generar un dataframe del valor absoluto
#de la base2. Esto se corrigio del codigo original 

#Predicción para controles
prediction = model.predict(Base_R1)
aciertos = 0
desaciertos = 0
for i in prediction:
    if i >=0.5:
        aciertos += 1
    else:
        desaciertos += 1
print(f'Los aciertos para controles son {aciertos}, los desaciertos para controles son {desaciertos}')

#Predicción para TDAH
prediction = model.predict(Base_R2)
aciertos = 0
desaciertos = 0
for i in prediction:
    if i <=0.5:
        aciertos += 1
    else:
        desaciertos += 1
print(f'Los aciertos para TDAH son {aciertos}, los desaciertos para TDAH son {desaciertos}')
