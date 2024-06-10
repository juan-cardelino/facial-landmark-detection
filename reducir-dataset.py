import os
import json
import numpy as np
import cv2

# Cortar imagen en base a boundingnbox, porcentaje achica el recorte
def cropp(frame, boundingbox, porcentaje = 0.2):
    offset_x = int(boundingbox[1][0]-boundingbox[0][0])
    offset_y = int(boundingbox[1][1]-boundingbox[0][1])
    
    start_x = boundingbox[0][0]+ int(offset_x*porcentaje)
    end_x = boundingbox[1][0]- int(offset_x*porcentaje)
    start_y = boundingbox[0][1]+ int(offset_y*porcentaje)
    end_y = boundingbox[1][1]- int(offset_y*porcentaje)
    return frame[start_y:end_y, start_x:end_x]

def color(frame):
    aux = np.mean(np.mean(frame, axis=0), axis=0)
    return [int(aux[0]), int(aux[1]), int(aux[2])]

# Lista de json generados de FFHQ
archivos = os.listdir("FFHQ Json")
# Direccion de las imagenes de FFHQ
FFHQ_path = "C:/Users/mauri/OneDrive/Desktop/Facultad/Cuarto/Pasantia/FFHQsmall/faces_dataset_small/"

# Lista de las imagenes que fueron procesadas de FFHQ
detection = []

# Separar detecciones del resto de los archivos
for i in archivos:
    # Se identifica si el json es de deteccion, si es se continua
    if i[-14:] == "deteccion.json":
        aux = i[:-15]
        while aux[0] == '0':
            aux = aux[1:]
        #print(aux)
        detection.append(aux)

# abrir dataset completo de FFHQ
with open('ffhq-dataset-v2.json') as dataset:
    ffhq_data = json.load(dataset)      

# Generar nuevo dataset con direccion de folder de fotos
new_ffhq_data = {"FFHQ_path": FFHQ_path}

for i in detection:
    print(i)
    # Extraer landmarks de dataset completo
    aux = np.array(ffhq_data[i]["image"]["face_landmarks"]).T
    # Calcular boundingbox a partir de landmarks
    boundingbox = [[int(min(aux[0])), int(min(aux[1]))], [int(max(aux[0])), int(max(aux[1]))]]

    # Se cacular el nombre de la imagen a partir de i y se lo guarda como j
    j = (5-len(i))*'0'+i
    
    # Leer fotos
    frame = cv2.imread(FFHQ_path+j+".png")
    
    # Cortar foto segun boundingbox
    frame_cropped = cropp(frame, boundingbox)
    
    # Color de imagen total
    frame_color = color(frame)
    # Color de cara
    face_color = color(frame_cropped)
    
    # Nueva entrada en nuevo dataset
    new_ffhq_data[i]={'country':ffhq_data[i]['metadata']['country'], 
                      'pixel_size':ffhq_data[i]['image']['pixel_size'],
                      'boundingbox': boundingbox, 
                      'image color': frame_color,
                      'face color': face_color}

# Guardar nuevo dataset
with open('ffhq-dataset-v3.json', 'w') as file:
    json.dump(new_ffhq_data, file, indent=4)