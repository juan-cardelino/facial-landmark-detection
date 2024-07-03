import os
import json
import numpy as np
from sklearn import metrics

# Direccion de jsons
dir_json = 'FFHQ Json'

# Direccion de las imagenes de FFHQ
FFHQ_path = 'FFHQ small'

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

# Inicio bloque identificar archivos con caras no detectadas
no_data = []

for i in detection:
    if not i in data:
        no_data.append(i)
# Fin bloque identificar archivos con caras no detectadas

information = []

for i in detection:
    with open('FFHQ_json_features/{}.json'.format(i)) as dataset:
        ffhq_data = json.load(dataset)
    if len(ffhq_data)>0:
        information.append([i in data, ffhq_data[0]['faceAttributes']['headPose']['pitch'], ffhq_data[0]['faceAttributes']['headPose']['roll'], ffhq_data[0]['faceAttributes']['headPose']['yaw']])

actual = np.array(information).T[0]
pitch = np.array(information).T[1]
roll = np.array(information).T[2]
yaw = np.array(information).T[3]

pitch_accuracy = []

for i_pitch in range(int(max(np.abs(pitch)))):
    predicted_pitch = (pitch < i_pitch)*1
    tn, fp, fn, tp = metrics.confusion_matrix(actual, predicted_pitch).ravel()
    pitch_accuracy.append([i_pitch, (tn+tp)*100/(tn+fp+fn+tp)])

roll_accuracy = []

for i_roll in range(int(max(np.abs(roll)))):
    predicted_roll = (roll < i_roll)*1
    tn, fp, fn, tp = metrics.confusion_matrix(actual, predicted_roll).ravel()
    roll_accuracy.append([i_roll, (tn+tp)*100/(tn+fp+fn+tp)])

yaw_accuracy = []

for i_yaw in range(int(max(np.abs(yaw)))):
    predicted_yaw = (yaw < i_yaw)*1
    tn, fp, fn, tp = metrics.confusion_matrix(actual, predicted_yaw).ravel()
    yaw_accuracy.append([i_yaw, (tn+tp)*100/(tn+fp+fn+tp)])

headpose_accuracy = []

for i_pitch in range(int(max(np.abs(pitch)))):
    print(i_pitch)
    for i_roll in range(int(max(np.abs(roll)))):
        for i_yaw in range(int(max(np.abs(yaw)))):
            predicted_pitch = (pitch < i_pitch)*1
            predicted_roll = (roll < i_roll)*1
            predicted_yaw = (yaw < i_yaw)*1
            predicted_headpose = []
            for i in range(len(predicted_pitch)):
                predicted_headpose.append(predicted_pitch[i]*predicted_roll[i]*predicted_yaw[i])
            tn, fp, fn, tp = metrics.confusion_matrix(actual, predicted_headpose).ravel()
            headpose_accuracy.append([i_pitch, i_roll, i_yaw, (tn+tp)*100/(tn+fp+fn+tp)])
            
            

print(i_pitch)
print(max(pitch_accuracy))
print(max(np.array(pitch_accuracy).T[1]))

print(i_roll)
print(max(roll_accuracy))
print(max(np.array(roll_accuracy).T[1]))

print(i_yaw)
print(max(yaw_accuracy))
print(max(np.array(yaw_accuracy).T[1]))

print(i_yaw)
print(max(headpose_accuracy))
print(max(np.array(headpose_accuracy).T[3]))
            
      

    


