import file_landmark as fland
import os
import procesamiento as pr
import json

# Initial setup
with open('configuracion.json') as file:
    configuracion = json.load(file)

verbose = configuracion["pipeline"]["from_file"]["verbose"]
minimo_ancho_de_cara = configuracion["pipeline"]["from_file"]["minimo_ancho_de_cara"]
etapas = configuracion["pipeline"]["from_file"]["etapas"]
raw_input = configuracion["path"]["input_dir"]
detected_output = configuracion["path"]["detect_dir"]
json_dir = configuracion["path"]["json_dir"]
json_suffix_detect = configuracion["path"]["json_suffix_detect"]
json_suffix_data = configuracion["path"]["json_suffix_data"]


archivos = os.listdir(raw_input)

print("Se corren {} de 3 etapas".format(etapas))

# Stage 1, get facial landmarks
if etapas > 0:
    print("\nInicio etapa 1")
    fland.find_landmarks(archivos, minimo_ancho_de_cara, verbose, raw_input, detected_output, json_dir, json_suffix_detect)
    print("Fin etapa 1\n")
    
# Stage 2, calculate facial feature from landmarks
if etapas > 1:
    print("Inicio etapa 2")
    imagenes = []
    for i in os.listdir("Json"):
        # Detect json suffix
        if i[-14:] == json_suffix_detect+'.json':
            imagenes.append(i[:-15])
    max_caras = 1

    for imagen in imagenes:
        # Get landmark from json
        ojoder, ojoizq, frente, boca, boundingbox, cant_caras = pr.carga_marcadores(imagen, max_caras = max_caras)

        for i in range(cant_caras):
            # Calculate facial features
            centroideder, centroideizq, unidad, origen_ojo, distojos, distfrente_ojo, distboca_ojo, angulo_cara, angulo_ojo_derecho, angulo_ojo_izquierdo, valores_elipse_ojoder, valores_elipse_ojoizq = pr.calculos(ojoder[i], ojoizq[i], frente[i], boca[i])
            # eje_ojos, p_eje_ojos, centrofrente, centroboca = pr.calculos_alter(ojoder[i], ojoizq[i], frente[i], boca[i])

            # Structured storage
            if i == 0:
                pr.guardar_marcadores(centroideder, centroideizq, unidad, origen_ojo, distojos, distfrente_ojo, distboca_ojo, angulo_cara, angulo_ojo_derecho, angulo_ojo_izquierdo, valores_elipse_ojoder, valores_elipse_ojoizq, boundingbox[0], imagen, json_dir, json_suffix_data)
                print('mejor cara guardada en '+imagen+'_'+json_suffix_data+'.json')

    print("Fin etapa 2\n")

# Stage 3, 
if etapas > 2:
    print("Inicio etapa 3\n")
    # Adapatar el codigo de cortar imagenes para meterlo
    print("Fin etapa 3\n")

print("Fin ejecucion")