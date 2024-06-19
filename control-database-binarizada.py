import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

def probar_caracteristica(ffhq_data, detection_with_feature, data_with_feature, caracteristica, verbose = False):
    
    actual = []
    predicted = []
    
    for i in detection_with_feature:
        actual.append(not i in data_with_feature)
        predicted.append(not ffhq_data[i]['data'][caracteristica])
    
    confusion_matrix = metrics.confusion_matrix(predicted, actual)
    tp = confusion_matrix[0][0]
    fp = confusion_matrix[0][1]
    fn = confusion_matrix[1][0]
    tn = confusion_matrix[1][1]

    if verbose:
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ["True", "False"])

        cm_display.plot()
        plt.ylabel(caracteristica)
        plt.xlabel("Cara detectada")
        plt.show()
    
    return [tp, fp, fn, tn]

def filtro(a, b, condicion=True):
    aux_True = []
    aux_False = []
    for i in a:
        if i in b:
            aux_True.append(i)
        else:
            aux_False.append(i)
    if condicion:
        return aux_True
    else:
        return aux_False

# Initial setup
with open('configuracion.json') as file:
    configuracion = json.load(file)

json_suffix_detect = configuracion["path"]["json_suffix_detect"]
json_suffix_data = configuracion["path"]["json_suffix_data"]
json_dir = configuracion['path']['json_dir']
dataset_binarizada_input = configuracion['pipeline']['binarizar dataset']['dataset_binarizada']
verbose = configuracion['pipeline']['binarizar dataset']['verbose']

# Aca reescribo porque si, en la version final borrarlo
dir_json = 'FFHQ Json'

# Lista de jsons
archivos_json = os.listdir(dir_json)

# Lista de imagenes procesadas
detection = []
# Lista de imagenes con caras
data = []

l_json_suffix_detect = len(json_suffix_detect)+5
l_json_suffix_data = len(json_suffix_data)+5

# Identificar json procesados y caras detectadas
for i in archivos_json:
    # Se identifica si el json es de deteccion
    if i[-l_json_suffix_detect:] == "{}.json".format(json_suffix_detect):
        detection.append(i[:-l_json_suffix_detect-1])
    # Se identifica si el json es de data
    elif i[-l_json_suffix_data:] == "{}.json".format(json_suffix_data):
        data.append(i[:-l_json_suffix_data-1])

if verbose:
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

with open('{}.json'.format(dataset_binarizada_input)) as archivo:
        ffhq_data = json.load(archivo)

no_existe_feature = ffhq_data['feature']['no existe']

detection_with_feature = filtro(detection, no_existe_feature, False)

data_with_feature = filtro(data, detection_with_feature, condicion=True)

if verbose:
    print('data without features: {}, {}%'.format(len(data)-len(data_with_feature), round(100*(len(data)-len(data_with_feature))/len(data), 2)))

no_data_with_feature = filtro(no_data, detection_with_feature, condicion=True)

if verbose:       
    print('no_data without features: {}, {}%'.format(len(no_data)-len(no_data_with_feature), round(100*(len(no_data)-len(no_data_with_feature))/len(no_data), 2)))

# Extraer claves del 
aux = list(ffhq_data.keys())
claves = list(ffhq_data[aux[1]]["data"].keys())

pruebas = []
for clave in claves:
    pruebas.append([clave, probar_caracteristica(ffhq_data, detection_with_feature, data_with_feature, clave, False)])

for prueba in pruebas:
    atributo = prueba[0]
    tp = prueba[1][0]
    fp = prueba[1][1]
    fn = prueba[1][2]
    tn = prueba[1][3]
    accuracy = (tp+tn)/(tp+fp+fn+tn)
    print('\n{}'.format(atributo.upper()))
    print('True Positive: {}\nFalse Positive: {}\nFalse Negative: {}\nTrue Negative: {}'.format(tp, fp, fn, tn))
    print("Accuracy: {}%".format(round(accuracy*100, 1)))

print('Fin ejecucion')