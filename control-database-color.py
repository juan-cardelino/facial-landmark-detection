import os
import json
import numpy as np
import matplotlib.pyplot as plt
import control_dataset

# Extract property from dataset, by default face color
def property(a, dataset, key = 'face color'):
    aux = []
    for i in a:
        aux.append(dataset[i][key])
    return np.array(aux).T

# Plot RGB colors
def plotBGR(a, b, title = ''):
    plt.figure()
    plt.title(title)
    plt.plot(a[0], a[1], "o")
    plt.plot(b[0], b[1], "x")
    plt.xlabel("Blue")
    plt.ylabel("Green")

    plt.figure()
    plt.title(title)
    plt.plot(a[1], a[2], "o")
    plt.plot(b[1], b[2], "x")
    plt.xlabel("Green")
    plt.ylabel("Red")

    plt.figure()
    plt.title(title)
    plt.plot(a[2], a[0], "o")
    plt.plot(b[2], b[0], "x")
    plt.xlabel("Red")
    plt.ylabel("Blue")
    return

print('\nCargando configuracion\n')

# Initial setup
with open('configuracion.json') as file:
    configuracion = json.load(file)

FFHQ_path = configuracion['path']['input_dir']
json_dir = configuracion['path']['json_dir']
json_suffix_detect = configuracion['path']['json_suffix_detect']
json_suffix_data = configuracion['path']['json_suffix_data']
dataset_input = configuracion['pipeline']['color dataset']['dataset_output']
verbose = configuracion['pipeline']['color dataset']['verbose']

# Aca reescribo porque si, en la version final borrarlo
FFHQ_path = 'FFHQ small'
json_dir = 'FFHQ Json'

print('Inicio ejecucion\n')

# List of FFHQ jsons
archivos_json = os.listdir(json_dir)

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

# Show amount of json
print("total json: {}".format(len(archivos_json)))
# Show amount of processed images
print("deteccion: {}".format(len(detection)))
# Show amount of images with faces
print("data: {}".format(len(data)))
# Show amount of images without faces and percentage from total
print("{} caras no detectadas, {}%\n".format(len(no_data), round(100*len(no_data)/len(detection), 2)))

# Open dataset json
with open('{}.json'.format(dataset_input)) as archivo:
        ffhq_data = json.load(archivo)

# Find indexes for the dataset
no_data_filtrada = control_dataset.reducir_0(no_data)
data_filtrada = control_dataset.reducir_0(data)

# Extract face color
face_color_no_data = property(no_data_filtrada, ffhq_data, 'face color')
face_color_data = property(data_filtrada, ffhq_data, 'face color')

# Extract frame color
frame_color_no_data = property(no_data_filtrada, ffhq_data, 'image color')
frame_color_data = property(data_filtrada, ffhq_data, 'image color')

if verbose:
    # Plot face color
    plotBGR(face_color_data, face_color_no_data, "face color")
    # Plot frame color
    plotBGR(frame_color_data, frame_color_no_data, "frame color")
    # Plot frma color - face color
    plotBGR(frame_color_data-face_color_data, frame_color_no_data-face_color_no_data, "frame-face color")

    plt.show()

# There was no way to predict if the face would be detected or not based on the color
