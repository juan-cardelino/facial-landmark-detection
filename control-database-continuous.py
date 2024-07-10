import os
import json
import matplotlib.pyplot as plt
import control_dataset

print('\nCargando configuracion\n')

# Initial setup
with open('configuracion.json') as file:
    configuration = json.load(file)

FFHQ_path = configuration['path']['input_dir']
json_dir = configuration['path']['json_dir']
dataset_dir = configuration['pipeline']['binarizar_dataset']['dataset_dir']
verbose = configuration['pipeline']['binarizar_dataset']['verbose']
json_suffix_detect = configuration['general']['json_suffix_detect']
json_suffix_data = configuration['general']['json_suffix_data']

# Aca reescribo porque si, en la version final borrarlo
FFHQ_path = 'FFHQ small'
json_dir = 'FFHQ Json'

print('Inicio ejecucion\n')

# List of FFHQ jsons
archivos = os.listdir(json_dir)

# List of processed images
detection = []
# List of images with faces
data = []

# Length of json soffix
l_json_suffix_detect = len(json_suffix_detect)+5
l_json_suffix_data = len(json_suffix_data)+5

# Load processed images
for i in archivos:
    # Check if json is from detection list
    if i[-l_json_suffix_detect:] == '{}.json'.format(json_suffix_detect):
        detection.append(i[:-l_json_suffix_detect-1])
    # Check if json is from data list
    elif i[-l_json_suffix_data:] == "{}.json".format(json_suffix_data):
        data.append(i[:-l_json_suffix_data-1])

pitch_detection = []
yaw_detection = []
roll_detection = []

pitch_data = []
yaw_data = []
roll_data = []

pitch_no_data = []
yaw_no_data = []
roll_no_data = []

for i in detection:
    # Open the feature.json of image
    with open('{}/{}.json'.format(dataset_dir, i)) as dataset:
        ffhq_data = json.load(dataset)
    
    if len(ffhq_data)>0:
        if i in data:
            pitch_data.append(ffhq_data[0]['faceAttributes']['headPose']['pitch'])
            yaw_data.append(ffhq_data[0]['faceAttributes']['headPose']['yaw'])
            roll_data.append(ffhq_data[0]['faceAttributes']['headPose']['roll'])
        else:
            pitch_no_data.append(ffhq_data[0]['faceAttributes']['headPose']['pitch'])
            yaw_no_data.append(ffhq_data[0]['faceAttributes']['headPose']['yaw'])
            roll_no_data.append(ffhq_data[0]['faceAttributes']['headPose']['roll'])
            
        pitch_detection.append(ffhq_data[0]['faceAttributes']['headPose']['pitch'])
        yaw_detection.append(ffhq_data[0]['faceAttributes']['headPose']['yaw'])
        roll_detection.append(ffhq_data[0]['faceAttributes']['headPose']['roll'])



fig, ax = plt.subplots()
plt.title('Pitch')
plt.hist(pitch_detection, bins=[-40, -30, -20, -10, 0, 10, 20, 30, 40])
fig, ax = plt.subplots()
plt.title('Yaw')
plt.hist(yaw_detection, bins=[-100, -75, -50, -25, 0, 25, 50, 75, 100])
fig, ax = plt.subplots()
plt.title('Roll')
plt.hist(roll_detection, bins=[-20, -15, -10, -5, 0, 5, 10, 15, 20])

plt.figure()
plt.plot(pitch_data, 'o')
plt.plot(pitch_no_data, 'x')

plt.figure()
plt.plot(yaw_data, 'o')
plt.plot(yaw_no_data, 'x')

plt.figure()
plt.plot(roll_data, 'o')
plt.plot(roll_no_data, 'x')

plt.show()