# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 16:34:38 2023

@author: abigu
Ya edité este archivo con mis propias direcciones.
"""

from sklearn.model_selection import train_test_split
import pandas as pd
import os
import numpy as np
import keras 
from keras.models import Sequential
from keras.layers import Dense, Dropout

#Controles
#Acá borré demasiadas líneas. Las pegaré en un documento aparte. PRIMERA PARTE BORRADA.
#Le borré los headers a mano en el excel. A ver qué pasa.
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
Base_R1_controles=Base_R1
listaa=[] #lista vacia

'''Cambiaría la forma en cómo se llena la lista con unos o ceros. Acá utilizó un ciclo, pero creo que consume muchos menos recursos hacerlo de otra forma.'''
for uno1 in range(len(Base_R1_controles.axes[0])):
    listaa.append(1)
#print(listaa)    
Base_R1_controles['y']=listaa
#print(Base_R1_controles)

#1=controles
Errores_1=Base_R2 #los errores son las base de tdah
listaa=[]

for ceroo in range(len(Errores_1.axes[0])):
    listaa.append(0)
#0=tdah    
Errores_1['y']=listaa
Base_R1_controles=Base_R1_controles.append(Errores_1,ignore_index=True)
val_fal_1=Base_R1_controles.isnull().sum()

y_1=Base_R1_controles.y 
#allY1=y_1
#allY1.append(0)
X_1=Base_R1_controles.drop('y',axis=1)



#TDAH 
#se crea una lista vacia para llenar con los errores y los errores se 
#etiquetan con 'y'
'''Base_R2_TDAH=pd.DataFrame(Base_R2)

listaa=[]
for uno1 in range(len(Base_R2_TDAH.axes[0])):
    listaa.append(1)
Base_R2_TDAH['y']=listaa
#1=tdah
Errores_2=Base_R1
listaa=[]

for ceroo in range(len(Errores_2.axes[0])):
    listaa.append(0)
Errores_2['y']=listaa
#0=controles
Base_R2_TDAH = Base_R2_TDAH.append(Errores_2,ignore_index=True)
val_fal_2=Base_R2_TDAH.isnull().sum()

y_2=Base_R2_TDAH.y
#allY2=y_2
#allY2.append(1)
X_2=Base_R2_TDAH.drop('y',axis=1)'''


#############################################

#X = X_1.append(X_2) #se juntan las 'x'
X = np.array(X_1) #se convierten en una matriz nunpy para que entre a la red 
#print(X)

#Y = y_1.append(y_2)
Y = np.array(y_1)
#print(Y)


#print(len(X))
#print(len(Y))



x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1) #Parece que esta función los revuelve de una vez...
#dfx = pd.DataFrame(x_train)
#dfx.to_csv('x_train_modified_2.csv', header = False, index = False)
#dfy = pd.DataFrame(y_train)
#dfy.to_csv('y_train_modified_2.csv', header = False, index = False)

#print(f'Esto es x_train\n{x_train}\nesto es y_train\n{y_train}')
#print(type(x_train))
#print(type(y_train))

opt = keras.optimizers.RMSprop(learning_rate=0.001)

model = Sequential()
model.add(Dense(len(x_train[1]), activation='sigmoid', input_shape=(len(x_train[1]),)))
model.add(Dropout(0.2))
#model.add(Dense(512, activation='sigmoid'))
#model.add(Dense(512, activation='sigmoid'))
#model.add(Dropout(0.2))
#model.add(Dense(512, activation='sigmoid'))
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy', optimizer= opt, metrics=("accuracy"))

history=model.fit(x_train, y_train, batch_size=(32),steps_per_epoch=150, epochs=60, verbose=1, 
                  validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print(score)

#Vamos a guardar la red para probarla con nuevos datos.
os.chdir(r'C:\Users\jairm\Downloads')
model.save("red_bin_2_epocas_entrenando_mas_datos_v4_60_epocas.h5")