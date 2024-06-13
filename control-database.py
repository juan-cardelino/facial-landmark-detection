import os
import json
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
print("{} caras no detectadas, {}%".format((len(detection)-len(data)), round(100*(len(detection)-len(data))/len(detection), 2)))

# Inicio bloque identificar archivos con caras no detectadas
iter_detected = 0
iter_data = 0
no_data = []

# Se corre un bucle while hasta que el iterador sea mayor o igual al largo de deteccioes
while iter_detected < len(detection) and iter_data < len(data):
    # Si el archivo existe en deteccion y data simultaneamente, se sigue con el siguiente
    if detection[iter_detected] == data[iter_data]:
        iter_data = iter_data+1
    # si no, se guarda la deteccion en no data
    else:
        no_data.append(detection[iter_detected])
    iter_detected = iter_detected+1
    
# Fin bloque identificar archivos con caras no detectadas
   
#print(len(no_detected))

# Abrir data set
with open('ffhq-dataset-v3.json') as archivo:
        ffhq_data = json.load(archivo)

# Buscar indices para el data set    
no_data_filtrada = filtado(no_data)
data_filtrada = filtado(data)

# Extraer color cara
face_color_no_data = color(no_data_filtrada, ffhq_data)
face_color_data = color(data_filtrada, ffhq_data)

# Extraer color frame
frame_color_no_data = color(no_data_filtrada, ffhq_data, 'image color')
frame_color_data = color(data_filtrada, ffhq_data, 'image color')

if 0:
    # Plotear color cara
    plotBGR(face_color_data, face_color_no_data, "face color")
    # Plotear color frame
    plotBGR(frame_color_data, frame_color_no_data, "frame color")
    # Plotear color frame - color cara
    plotBGR(frame_color_data-face_color_data, frame_color_no_data-face_color_no_data, "frame-face color")

    plt.show()

# No se encontro una forma de predecir si la cara va a ser bien o mal detectada en base al color