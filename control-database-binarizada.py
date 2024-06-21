import os
import json
import matplotlib.pyplot as plt
from sklearn import metrics
import control_dataset

def probar_caracteristica(ffhq_data, detection_with_feature, data_with_feature, caracteristica, verbose = False):
    
    # List of detected and not detected
    actual = []
    # List of predictions
    predicted = []
    
    # Load lists actual and predicted
    for i in detection_with_feature:
        actual.append(not i in data_with_feature)
        predicted.append(not ffhq_data[i]['data'][caracteristica])
    
    # Calculate confusion matrix
    confusion_matrix = metrics.confusion_matrix(predicted, actual)
    # Extract True Positive, False positive, False negative and True negative from confusion matrix
    tp = confusion_matrix[0][0]
    fp = confusion_matrix[0][1]
    fn = confusion_matrix[1][0]
    tn = confusion_matrix[1][1]

    # Show confusion matrix
    if verbose:
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ["True", "False"])

        cm_display.plot()
        plt.ylabel(caracteristica)
        plt.xlabel("Cara detectada")
        plt.show()
    
    # Return True Positive, False Positive, False Negative and True Negative
    return [tp, fp, fn, tn]

print('\nCargando configuracion\n')

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

print('Inicio ejecucion\n')

# List of jsons
archivos_json = os.listdir(dir_json)

# List of processed images
detection = []
# List of images with faces
data = []

# Length of json soffix
l_json_suffix_detect = len(json_suffix_detect)+5
l_json_suffix_data = len(json_suffix_data)+5

# Load processed images and images with faces lists
for i in archivos_json:
    # Check if json is from detection list
    if i[-l_json_suffix_detect:] == "{}.json".format(json_suffix_detect):
        detection.append(i[:-l_json_suffix_detect-1])
    # Check if json is from data list
    elif i[-l_json_suffix_data:] == "{}.json".format(json_suffix_data):
        data.append(i[:-l_json_suffix_data-1])

# List of images without faces
no_data = control_dataset.filtro(detection, data, False)

if verbose:
    # Show amount of json
    print("total json: {}".format(len(archivos_json)))
    # Show amount of processed images
    print("deteccion: {}".format(len(detection)))
    # Show amount of images with faces
    print("data: {}".format(len(data)))
    # Show amount of images without faces and percentage from total
    print("{} caras no detectadas, {}%\n".format(len(no_data), round(100*len(no_data)/len(detection), 2)))

# Load dataset json
with open('{}.json'.format(dataset_binarizada_input)) as archivo:
        ffhq_data = json.load(archivo)

# List of images without features
no_existe_feature = ffhq_data['feature']['no existe']

# List of processed images with features
detection_with_feature = control_dataset.filtro(detection, no_existe_feature, False)

# List of images with faces and features
data_with_feature = control_dataset.filtro(data, detection_with_feature, True)

# Show amount of images with faces and features
if verbose:
    print('data without features: {}, {}%'.format(len(data)-len(data_with_feature), round(100*(len(data)-len(data_with_feature))/len(data), 2)))

# List of images without faces but with features
no_data_with_feature = control_dataset.filtro(no_data, detection_with_feature, True)

# Show amount of images without faces but with features
if verbose:       
    print('no_data without features: {}, {}%'.format(len(no_data)-len(no_data_with_feature), round(100*(len(no_data)-len(no_data_with_feature))/len(no_data), 2)))

# Extract keys from dataset json
aux = list(ffhq_data.keys())
claves = list(ffhq_data[aux[1]]["data"].keys())

# List of attribute confusion matrix
pruebas = []
# Load attribute confusion matrix list
for clave in claves:
    pruebas.append([clave, probar_caracteristica(ffhq_data, detection_with_feature, data_with_feature, clave, False)])

for prueba in pruebas:
    # Extract attribute
    atributo = prueba[0]
    # Extrar confusion matrix
    tp = prueba[1][0]
    fp = prueba[1][1]
    fn = prueba[1][2]
    tn = prueba[1][3]
    # Calculate accuracy
    accuracy = (tp+tn)/(tp+fp+fn+tn)
    # Show attribute
    print('\n{}'.format(atributo.upper()))
    # Show confusion matrix
    print('True Positive: {}\nFalse Positive: {}\nFalse Negative: {}\nTrue Negative: {}'.format(tp, fp, fn, tn))
    # Show accuracy
    print("Accuracy: {}%".format(round(accuracy*100, 1)))

print('\nFin ejecucion\n')