import os
import json
import numpy as np
import cv2
import alinear

def color(frame):
    aux = np.mean(np.mean(frame, axis=0), axis=0)
    return [int(aux[0]), int(aux[1]), int(aux[2])]

print('\nCargando configuracion\n')

# Initial setup
with open('configuracion.json') as file:
    configuracion = json.load(file)

FFHQ_path = configuracion['path']['input_dir']
json_dir = configuracion['path']['json_dir']
json_suffix_detect = configuracion['path']['json_suffix_detect']
dataset_input = configuracion['pipeline']['color dataset']['dataset_input']
dataset_output = configuracion['pipeline']['color dataset']['dataset_output']
verbose = configuracion['pipeline']['binarizar dataset']['verbose']

# Aca reescribo porque si, en la version final borrarlo
FFHQ_path = 'FFHQ small'
json_dir = 'FFHQ Json'

print('Inicio ejecucion\n')

# List of FFHQ jsons
archivos = os.listdir(json_dir)

# List of processed images
detection = []

# Length of json soffix
l_json_suffix_detect = len(json_suffix_detect)+5

# Load processed images
for i in archivos:
    # Check if json is from detection list
    if i[-l_json_suffix_detect:] == '{}.json'.format(json_suffix_detect):
        aux = i[:-l_json_suffix_detect-1]
        while aux[0] == '0':
            aux = aux[1:]
        detection.append(aux)

# Open full FFHQ dataset
with open('{}.json'.format(dataset_input)) as dataset:
    ffhq_data = json.load(dataset)      

# Generate new dataset with images folder direction
new_ffhq_data = {"FFHQ_path": FFHQ_path}

for i in detection:
    if int(i) % 100 == 0:
        print(i)
    # Extract landmarks from full FFHQ dataset
    landmarks = np.array(ffhq_data[i]["image"]["face_landmarks"]).T
    # calculate boundingbox from landmarks
    boundingbox = [int(min(landmarks.T[0])), int(min(landmarks.T[1])), int(max(landmarks.T[0])-min(landmarks.T[0])), int(max(landmarks.T[1])-min(landmarks.T[1]))]

    # Calculate image name from iamge number
    j = (5-len(i))*'0'+i
    
    # Read image
    frame = cv2.imread('{}/{}.png'.format(FFHQ_path, j))
    
    # Cropp image using boundingbox
    frame_cropped = alinear.cropp(frame, boundingbox, 0.2)
    
    # Calculate iamge color
    frame_color = color(frame)
    # Calculate face color
    face_color = color(frame_cropped)
    
    # Load information into new dataset
    new_ffhq_data[i]={'country':ffhq_data[i]['metadata']['country'], 
                      'pixel_size':ffhq_data[i]['image']['pixel_size'],
                      'boundingbox': boundingbox, 
                      'image color': frame_color,
                      'face color': face_color}

# Save new dataset
with open('{}.json'.format(dataset_output), 'w') as file:
    json.dump(new_ffhq_data, file, indent=4)