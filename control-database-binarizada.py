import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

def probar_caracteristica(ffhq_data, detection_with_feature, data_with_feature, caracteristica, verbose = False):
    
    actual = []
    predicted = []
    
    print(len(data_with_feature))
    
    for i in detection_with_feature:
        actual.append(not i in data_with_feature)
        predicted.append(not ffhq_data[i]['data'][caracteristica])
    
    confusion_matrix = metrics.confusion_matrix(predicted, actual)

    if verbose:
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ["True", "False"])

        cm_display.plot()
        plt.ylabel(caracteristica)
        plt.xlabel("Cara detectada")
        plt.show()
    
    return

# Direccion de jsons
dir_json = 'FFHQ Json'

# Lista de jsons
archivos_json = os.listdir(dir_json)

# Lista de imagenes procesadas
detection = []
# Lista de imagenes con caras
data = []

# Identificar json procesados y caras detectadas
for i in archivos_json:
    # Se identifica si el json es de deteccion
    if i[-14:] == "deteccion.json":
        detection.append(i[:-15])
    # Se identifica si el json es de data
    elif i[-9:] == "data.json":
        data.append(i[:-10])

# Se muestra la cantidad de jsons
print("total json: {}".format(len(archivos_json)))
# Se muestra la cantidad de imagenes procesadas
print("deteccion: {}".format(len(detection)))
# Se muestra la cantidad de imagenes con caras detectadas
print("data: {}".format(len(data)))
# Se muestra la cantidad de caras no detectadas y el porcentaje sobre el total
print("{} caras no detectadas, {}%\n".format((len(detection)-len(data)), round(100*(len(detection)-len(data))/len(detection), 2)))

# Inicio bloque identificar archivos con caras no detectadas
no_data = []

for i in detection:
    if not i in data:
        no_data.append(i)
# Fin bloque identificar archivos con caras no detectadas

with open('ffhq-feature-dataset-binarizado.json') as archivo:
        ffhq_data = json.load(archivo)

no_existe_feature = ffhq_data['feature']['no existe']

detection_with_feature = detection[:]
for i in no_existe_feature:
    if i in detection_with_feature:
        detection_with_feature.remove(i)

data_with_feature = data[:]
for i in data:
    if not i in  detection_with_feature:
        data_with_feature.remove(i)

print('data without features: {}, {}%'.format(len(data)-len(data_with_feature), round(100*(len(data)-len(data_with_feature))/len(data), 2)))

no_data_with_feature = no_data[:]
for i in no_data:
    if not i in  detection_with_feature:
        no_data_with_feature.remove(i)
        
print('no_data without features: {}, {}%'.format(len(no_data)-len(no_data_with_feature), round(100*(len(no_data)-len(no_data_with_feature))/len(no_data), 2)))

claves = list(ffhq_data["00055"]["data"].keys())
print(claves)
caracteristica = claves[6]

probar_caracteristica(ffhq_data, detection_with_feature, data_with_feature, caracteristica, True)

# sklearn o scipy-learn matriz de conjuncion 