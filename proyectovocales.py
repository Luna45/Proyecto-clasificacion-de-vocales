# Importar librerías necesarias para el proyecto
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import librosa # En Anaconda Promp: conda install -c conda-forge librosa
import csv
import os
import pandas as pd
from sklearn import preprocessing

#------------------------- EXTRACCIÓN Y PREPARACIÓN DE LOS DATOS ------------------------------------------------

# Las funciones utilizadas para esta sección "extraccionArchivosWAV" y "preProcessDatos" son adaptaciones de las funciones
# extractWavFeatures y preProcessData del proyecto Spoken Digits Recognition https://github.com/ravasconcelos/spoken-digits-recognition

# Función para convertir las características de los archivos WAV en datos en formato csv
def extraccionArchivosWAV(carpetaDeArchivos, ArchivoCSV):
    print("las características de los archivos en la carpeta "+carpetaDeArchivos+" se están guardando en el archivo "+ArchivoCSV)
    header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    for i in range(1, 21):
        header += f' mfcc{i}'
    header += ' label'
    header = header.split()
    #print('CSV Header: ', header)
    file = open(ArchivoCSV, 'w', newline='')
    writer = csv.writer(file)
    writer.writerow(header)
    #vocales = 'a e i o u'.split()
    for filename in os.listdir(carpetaDeArchivos):
        
        letra = f'{carpetaDeArchivos}/{filename}'
        y, sr = librosa.load(letra, mono=True, duration=30)
        
        # Quitar ruidos y silencios de los archivos
        y, index = librosa.effects.trim(y)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        rmse = librosa.feature.rms(y=y)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        mfcc = librosa.feature.mfcc(y=y, sr=sr)
        to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
        for e in mfcc:
            to_append += f' {np.mean(e)}'
        writer.writerow(to_append.split())
        
    file.close()
    print("Fin de extracción de los archivos")
    
# Función de preprocesamiento de los datos
def preProcessDatos(ArchivoCSV):
    print("procesando "+ArchivoCSV)
    data = pd.read_csv(ArchivoCSV)
    data['letra'] = data['filename'].str[:1]
    
    #Eliminando columnas innecesarias
    data = data.drop(['filename'],axis=1)
    data = data.drop(['label'],axis=1)
    data = data.drop(['chroma_stft'],axis=1)
    data.shape

    print("Los datos han sido preprocesados")
    return data


# Definir los nombre de los archivos csv
train_csv = "train.csv"
test_csv = "test.csv"

# Creación de archivos CSV
crear_csv = False # Si es True, los valores de los audios se guardaran en un archivo csv
                 # Como es una tarea que consume mucho tiempo, se dejará en False si ya se tienen los archivos

if (crear_csv == True):
    print("Extrayendo archivos de audio")
    extraccionArchivosWAV("sonidos vocales test", test_csv)
    extraccionArchivosWAV("sonidos vocales train", train_csv)
    print("Se han creado los archivos csv")
else:
    print("Se ha omitido la creación de archivos csv")
    
# Preprocesamiento y creación de datasets

trainData = preProcessDatos(train_csv)
testData = preProcessDatos(test_csv)

# -------------- ENTENDER LOS DATOS DE AUDIO Y HABLAR SOBRE COMO SE EXPRESAN LAS VOCALES ------------------


# Explicar que significa cada una de las características del audio
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

plt.figure()
for i in range(0,54):
    arreglo = np.array(trainData.iloc[[i]])
    arreglo = np.transpose(arreglo)
    arreglo = arreglo[6:24]
    arreglo = scaler.fit_transform(arreglo)
    x = np.linspace(0,18,18)
    plt.title("mfcc letra A")
    plt.plot(x,arreglo)
    
plt.figure()  
for i in range(55,108):
    arreglo = np.array(trainData.iloc[[i]])
    arreglo = np.transpose(arreglo)
    arreglo = arreglo[6:24]
    arreglo = scaler.fit_transform(arreglo)
    x = np.linspace(0,18,18)
    plt.title("mfcc letra E")
    plt.plot(x,arreglo)
    
plt.figure()   
for i in range(109,161):
    arreglo = np.array(trainData.iloc[[i]])
    arreglo = np.transpose(arreglo)
    arreglo = arreglo[6:24]
    arreglo = scaler.fit_transform(arreglo)
    x = np.linspace(0,18,18)
    plt.title("mfcc letra I")
    plt.plot(x,arreglo)

plt.figure()   
for i in range(162,215):
    arreglo = np.array(trainData.iloc[[i]])
    arreglo = np.transpose(arreglo)
    arreglo = arreglo[6:24]
    arreglo = scaler.fit_transform(arreglo)
    x = np.linspace(0,18,18)
    plt.title("mfcc letra O")
    plt.plot(x,arreglo)

plt.figure()
for i in range(216,269):
    arreglo = np.array(trainData.iloc[[i]])
    arreglo = np.transpose(arreglo)
    arreglo = arreglo[6:24]
    arreglo = scaler.fit_transform(arreglo)
    x = np.linspace(0,18,18)
    plt.title("mfcc letra U")
    plt.plot(x,arreglo)


# Also known as pitch class profile (PCP). Chroma representations measure the amount of relative energy in each pitch class (e.g., the 12 notes in the chromatic scale) at a given frame/time.

# Determinar características importantes en la clasificación de vocales y características que podemos omitir

# ----------------------------- DEFINIR DATOS DE VALIDACIÓN, TRAIN Y TEST ---------------------------------

from sklearn.model_selection import train_test_split
X = np.array(trainData.iloc[:, :-1], dtype = float)
y = trainData.iloc[:, -1]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

X_test = np.array(testData.iloc[:, :-1], dtype = float)
y_test = testData.iloc[:, -1]

# Normalización de los datos con preprocessing

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform( X_train ) # Datos de entrenamiento
X_val = scaler.transform( X_val ) # Datos de validación
X_test = scaler.transform( X_test ) # Datos de test

# ----------------------------------------- CREACIÓN DEL MODELO ---------------------------------------------
import keras
from keras import models
from keras import layers

# ------------------------------------------ TEST Y RESULTADOS ----------------------------------------------
# matriz de confusión
# Reporte de clasificación
# Accuracy - Performance
