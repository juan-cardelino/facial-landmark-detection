import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Extraer posicion en json a partir de nombre de imagen
def filtado(a):
    filtrada = []
    for i in a:
        aux = i
        while aux[0] == '0':
            aux = aux[1:]
        filtrada.append(aux)
    return filtrada

# Extraer propiedad de dataset, por defecto face color
def color(a, dataset, key = 'face color'):
    aux = []
    for i in a:
        aux.append(dataset[i][key])
    return np.array(aux).T

# Plotear los colores de RGB
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

# Direccion de jsons
dir_json = 'FFHQ Json'

archivos_json = os.listdir(dir_json)

detection = []
data = []

for i in archivos_json:
    # Se identifica si el json es de deteccion, si es se continua
    if i[-14:] == "deteccion.json":
        detection.append(i[:-15])
    elif i[-9:] == "data.json":
        data.append(i[:-10])

print("total json: {}".format(len(archivos_json)))
print("deteccion: {}".format(len(detection)))
print("data: {}".format(len(data)))
print("{} caras no detectadas, {}%".format((len(detection)-len(data)), round(100*(len(detection)-len(data))/len(detection), 2)))

iter_detected = 0
iter_data = 0
no_detected = []

while iter_detected < len(detection):
    if detection[iter_detected] == data[iter_data]:
        iter_data = iter_data+1
    else:
        no_detected.append(detection[iter_detected])
    iter_detected = iter_detected+1
        
#print(len(no_detected))

no_detected_filtrada = filtado(no_detected)
data_filtrada = filtado(data)

# Abrir data set
with open('ffhq-dataset-v3.json') as archivo:
        ffhq_data = json.load(archivo)

face_color_no_detected = color(no_detected_filtrada, ffhq_data)
face_color_data = color(data_filtrada, ffhq_data)

frame_color_no_detected = color(no_detected_filtrada, ffhq_data, 'image color')
frame_color_data = color(data_filtrada, ffhq_data, 'image color')

plotBGR(face_color_data, face_color_no_detected, "face color")

plotBGR(frame_color_data, frame_color_no_detected, "frame color")

plotBGR(frame_color_data-face_color_data, frame_color_no_detected-face_color_no_detected, "frame-face color")

plt.show()

# No se encontro una forma de predecir si la cara va a ser bien o mal detectada en base al color