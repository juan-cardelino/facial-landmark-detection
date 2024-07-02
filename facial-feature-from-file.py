import os
import json
import file_landmark as fland
import procesamiento as pr
import alinear
import cv2

# Initial setup
with open('configuracion.json') as file:
    configuracion = json.load(file)

verbose = configuracion["pipeline"]["from_file"]["verbose"]
minimo_ancho_de_cara = configuracion["pipeline"]["from_file"]["minimo_ancho_de_cara"]
etapas = configuracion["pipeline"]["from_file"]["etapas"]
raw_input = configuracion["path"]["input_dir"]
detected_output = configuracion["path"]["detect_dir"]
aligned_output = configuracion["path"]["aligned_dir"]
model_dir = configuracion["path"]["model_dir"]
json_dir = configuracion["path"]["json_dir"]
json_suffix_detect = configuracion["path"]["json_suffix_detect"]
json_suffix_data = configuracion["path"]["json_suffix_data"]
resize = configuracion["general"]["resize"]


archivos = os.listdir(raw_input)

print("Se corren {} de 3 etapas".format(etapas))

# Stage 1, get facial landmarks
if etapas > 0:
    print("\nInicio etapa 1")
    fland.find_landmarks(archivos, minimo_ancho_de_cara, verbose, raw_input, detected_output, json_dir, json_suffix_detect, model_dir, resize)
    print("Fin etapa 1\n")
    
# Stage 2, calculate facial feature from landmarks
if etapas > 1:
    print("Inicio etapa 2")
    imagenes = []
    l_suffix = len(json_suffix_detect)+5
    for i in os.listdir(json_dir):
        # Detect json suffix
        if i[-l_suffix:] == json_suffix_detect+'.json':
            imagenes.append(i[:-l_suffix-1])
    max_caras = 1

    for imagen in imagenes:
        # Get landmark from json
        imagen_file, ojoder, ojoizq, frente, boca, boundingbox, cant_caras = pr.load_landmarks(imagen, max_caras, json_dir)

        for i in range(cant_caras):
            # Calculate facial features
            centroideder, centroideizq, unidad, origen_ojo, distojos, distfrente_ojo, distboca_ojo, angulo_cara, angulo_ojo_derecho, angulo_ojo_izquierdo, valores_elipse_ojoder, valores_elipse_ojoizq = pr.calculate_facial_feature(ojoder[i], ojoizq[i], frente[i], boca[i])

            # Structured storage
            if i == 0:
                pr.save_features(imagen_file, centroideder, centroideizq, unidad, origen_ojo, distojos, distfrente_ojo, distboca_ojo, angulo_cara, angulo_ojo_derecho, angulo_ojo_izquierdo, valores_elipse_ojoder, valores_elipse_ojoizq, boundingbox[0], imagen, json_dir, json_suffix_data)
                print('mejor cara guardada en {}_{}.json'.format(imagen, json_suffix_data))

    print("Fin etapa 2\n")

# Stage 3, 
if etapas > 2:
    print("Inicio etapa 3\n")
    # Finding images with faces
    # Length of json suffix
    l_suffix = len(json_suffix_data)+5
    # List of images
    datas = []
    for i in os.listdir(json_dir):
        # Detect json suffix
        if i[-l_suffix:] == json_suffix_data+'.json':
            # Append image without suffix
            datas.append(i[:-l_suffix-1])
    
    # Cycle through image extension
    for data in datas:
        # Boundingbox and angle from data.json
        image_file, boundingbox, angulo = alinear.get_json_data(data, json_dir, json_suffix_data)
        print(image_file)
        # Get frame
        frame = cv2.imread('{}/{}'.format(raw_input, image_file))
        # Rotate frame using boundingbox
        frame_rotated = alinear.rotate(frame, boundingbox, angulo)
        # Cropp frame using boundingbox
        frame_cropped = alinear.cropp(frame_rotated, boundingbox)
        # Save frame in aligned folder
        cv2.imwrite('{}/{}.jpg'.format(aligned_output, data), frame_cropped)
        print('{}.jpg guardada en {} folder'.format(data, aligned_output))
    print("\nFin etapa 3\n")

print("Fin ejecucion")