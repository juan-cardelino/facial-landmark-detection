'''
This program takes the FFHQ feature metadata, extracts the pose of the face and looks for an umbral to predict if an image will be well detected based on the pose
'''

import os
import json
import numpy as np
from sklearn import metrics
import control_dataset

def main():
    '''
    Run program:
    
    Stars by loading feature FFHQ dataset
    
    Then calculates how many images were correctly detected
    
    Then extracts the headpose metadata
    
    Then trys many umbrals for the binarization of the data and calculates it confusion matrix
    
    Finally use the confusion matrix acurracy to determine wich umbral is the best and prints it in console
    '''
    print('\nCargando configuracion\n')

    # Initial setup
    with open('configuracion.json') as file:
        configuration = json.load(file)

    FFHQ_path = configuration['path']['input_dir']
    json_dir = configuration['path']['json_dir']
    dataset_dir = configuration['pipeline']['binarizar_dataset']['dataset_dir']
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

    # List of images without faces
    no_data = control_dataset.intersection(detection, data, False)

    information = []

    for i in detection:
        with open('{}/{}.json'.format(dataset_dir, i)) as dataset:
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
    return

if __name__ == "__main__":
    main()