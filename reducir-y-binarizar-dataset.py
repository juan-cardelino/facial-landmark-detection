import os
import json
import numpy as np

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
        detection.append(i[:-15])

# Generar nuevo dataset con direccion de folder de fotos
new_ffhq_data = {"FFHQ_path": FFHQ_path}
existe_json = []
no_existe_json = []

for i in detection:
    with open('FFHQ_json_features/{}.json'.format(i)) as dataset:
        ffhq_data = json.load(dataset)
    new_ffhq_data[i] = {}
    if len(ffhq_data)>0:
        new_ffhq_data[i]['error'] = 'no'
        new_ffhq_data[i]['data'] = {}
        new_ffhq_data[i]['data']['positove headpitch'] = ffhq_data[0]['faceAttributes']['headPose']['pitch'] > 0
        new_ffhq_data[i]['data']['positive headroll'] = ffhq_data[0]['faceAttributes']['headPose']['roll'] > 0
        new_ffhq_data[i]['data']['positive headyaw'] = ffhq_data[0]['faceAttributes']['headPose']['yaw'] > 0
        new_ffhq_data[i]['data']['gender_male'] = ffhq_data[0]['faceAttributes']['gender'] == 'male'
        new_ffhq_data[i]['data']['not glasses'] = not ffhq_data[0]['faceAttributes']['glasses'] == 'Glasses'
        new_ffhq_data[i]['data']['not blur'] = ffhq_data[0]['faceAttributes']['blur']['blurLevel'] == "low"
        new_ffhq_data[i]['data']['not noise'] = ffhq_data[0]['faceAttributes']['noise']['noiseLevel'] == "low"
        new_ffhq_data[i]['data']['forehead not occluded'] = not ffhq_data[0]['faceAttributes']['occlusion']['foreheadOccluded']
        new_ffhq_data[i]['data']['eye not occluded'] = not ffhq_data[0]['faceAttributes']['occlusion']['eyeOccluded']
        new_ffhq_data[i]['data']['mouth not occluded'] = not ffhq_data[0]['faceAttributes']['occlusion']['mouthOccluded']
        existe_json.append(i)
    else:
        new_ffhq_data[i]['error'] = 'si'
        no_existe_json.append(i)

new_ffhq_data['feature'] = {'existe':existe_json, 'no existe': no_existe_json}

print(len(detection))
print(len(new_ffhq_data)-1)

with open('ffhq-feature-dataset-binarizado.json', 'w') as file:
    json.dump(new_ffhq_data, file, indent=4)

