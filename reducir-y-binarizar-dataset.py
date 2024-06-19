import os
import json
import numpy as np

# Initial setup
with open('configuracion.json') as file:
    configuracion = json.load(file)

FFHQ_path = configuracion['path']['input_dir']
json_dir = configuracion['path']['json_dir']
json_suffix_detect = configuracion['path']['json_suffix_detect']
dataset_dir = configuracion['pipeline']['binarizar dataset']['dataset_dir']
dataset_binarizada_output = configuracion['pipeline']['binarizar dataset']['dataset_binarizada']

# Aca reescribo porque si, en la version final borrarlo
FFHQ_path = 'FFHQ small'
json_dir = 'FFHQ Json'

# Lista de json generados de FFHQ
archivos = os.listdir(json_dir)

# Lista de las imagenes que fueron procesadas de FFHQ
detection = []

# Separar detecciones del resto de los archivos
l_json_suffix_detect = len(json_suffix_detect)+5
for i in archivos:
    # Se identifica si el json es de deteccion, si es se continua
    if i[-l_json_suffix_detect:] == '{}.json'.format(json_suffix_detect):
        detection.append(i[:-l_json_suffix_detect-1])

# Generar nuevo dataset con direccion de folder de fotos
new_ffhq_data = {"FFHQ_path": FFHQ_path}
# Lista de imagenes con feature
existe_json = []
# Lista de imagenes sin feature
no_existe_json = []

# Condiciones de pitch, roll y yaw
pitch_condition = 20
roll_condition = 12
yaw_condition = 75

# Control de atributos
control_atributo = []

for i in detection:
    # Abrir el feature.json correspondiente a la imagen
    with open('{}/{}.json'.format(dataset_dir, i)) as dataset:
        ffhq_data = json.load(dataset)
    # Crear un atributo en el diccionario para guardar caracteristicas de la imagen
    new_ffhq_data[i] = {}
    # Verificar si feature.json tiene informacion o no. Los feature.json so listas con diccionarios adentro
    if len(ffhq_data)>0:
        # Guardar que feature.json tiene informacion
        new_ffhq_data[i]['error'] = False
        # Generar atributo data para guardar la informacion relevante
        new_ffhq_data[i]['data'] = {}
        
        # Este bloque extrae las caracteristicas que son relevantes y la binariza
        new_ffhq_data[i]['data']['positove headpitch'] = ffhq_data[0]['faceAttributes']['headPose']['pitch'] > 0
        new_ffhq_data[i]['data']['positive headroll'] = ffhq_data[0]['faceAttributes']['headPose']['roll'] > 0
        new_ffhq_data[i]['data']['positive headyaw'] = ffhq_data[0]['faceAttributes']['headPose']['yaw'] > 0
        new_ffhq_data[i]['data']['centred headpitch'] = ffhq_data[0]['faceAttributes']['headPose']['pitch'] < pitch_condition and ffhq_data[0]['faceAttributes']['headPose']['pitch'] > -pitch_condition
        new_ffhq_data[i]['data']['centred headroll'] = ffhq_data[0]['faceAttributes']['headPose']['roll'] < roll_condition and ffhq_data[0]['faceAttributes']['headPose']['roll'] > -roll_condition
        new_ffhq_data[i]['data']['centred headyaw'] = ffhq_data[0]['faceAttributes']['headPose']['yaw'] < yaw_condition and ffhq_data[0]['faceAttributes']['headPose']['yaw'] > -yaw_condition
        new_ffhq_data[i]['data']['centred face'] = new_ffhq_data[i]['data']['centred headpitch'] and new_ffhq_data[i]['data']['centred headroll'] and new_ffhq_data[i]['data']['centred headyaw']
        new_ffhq_data[i]['data']['gender male'] = ffhq_data[0]['faceAttributes']['gender'] == 'male'
        new_ffhq_data[i]['data']['not glasses'] = ffhq_data[0]['faceAttributes']['glasses'] == 'NoGlasses'
        new_ffhq_data[i]['data']['not blur'] = ffhq_data[0]['faceAttributes']['blur']['blurLevel'] == "low"
        new_ffhq_data[i]['data']['not noise'] = ffhq_data[0]['faceAttributes']['noise']['noiseLevel'] == "low"
        new_ffhq_data[i]['data']['forehead not occluded'] = not ffhq_data[0]['faceAttributes']['occlusion']['foreheadOccluded']
        new_ffhq_data[i]['data']['eye not occluded'] = not ffhq_data[0]['faceAttributes']['occlusion']['eyeOccluded']
        new_ffhq_data[i]['data']['mouth not occluded'] = not ffhq_data[0]['faceAttributes']['occlusion']['mouthOccluded']
        # Fin bloque
        
        # Guardar imagen en lista de archivos con feature
        existe_json.append(i)
        
        # Este bloque existe con el fin de hacer pruebas
        # Devuelve una lista con lo los posibles valores de un tributo en especifico de feature.json
        atributo = ffhq_data[0]['faceAttributes']['headPose']['yaw']
        if not atributo in control_atributo:
            control_atributo.append(atributo)
        # Fin bloque
    else:
        # Se guarda que feature.json no tiene informacion
        new_ffhq_data[i]['error'] = True
        # Guardar imagen en lista de archivos sin feature
        no_existe_json.append(i)

# Se guarla la lista de archivos con y sin feature en el nuevo json
new_ffhq_data['feature'] = {'existe':existe_json, 'no existe': no_existe_json}

# Guardar dataset reducido y binarizado
print('Generando dataset reducido y binarizado')
with open('{}.json'.format(dataset_binarizada_output), 'w') as file:
    json.dump(new_ffhq_data, file, indent=4)

if 0:
    print(min(control_atributo), max(control_atributo))
    
print('Fin ejecucion')