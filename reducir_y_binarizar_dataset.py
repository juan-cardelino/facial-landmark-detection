'''
This program takes the feature FFHQ dataset. It returns a json with a selection of feature FFHQ metadata binarized
'''
import os
import json
import numpy as np

def main():
    '''
    Rune code
    '''
    print('\nCargando configuracion\n')

    # Initial setup
    with open('configuracion.json') as file:
        configuration = json.load(file)

    FFHQ_path = configuration['path']['input_dir']
    json_dir = configuration['path']['json_dir']
    dataset_dir = configuration['pipeline']['binarizar_dataset']['dataset_dir']
    dataset_binarizada_output = configuration['pipeline']['binarizar_dataset']['dataset_binarizada']
    verbose = configuration['pipeline']['binarizar_dataset']['verbose']
    json_suffix_detect = configuration['general']['json_suffix_detect']

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
            detection.append(i[:-l_json_suffix_detect-1])

    # Generate new dataset with images folder direction
    new_ffhq_data = {"FFHQ_path": FFHQ_path}
    # List of images with features
    existe_json = []
    # List of images without features
    no_existe_json = []

    # Conditions of pitch, roll and yaw
    pitch_condition = 20
    roll_condition = 12
    yaw_condition = 75

    # Key control
    control_clave = []

    for i in detection:
        # Open the feature.json of image
        with open('{}/{}.json'.format(dataset_dir, i)) as dataset:
            ffhq_data = json.load(dataset)
        # Create image key
        new_ffhq_data[i] = {}
        # Check if feature.json has information or not. The feature.json are lists with dictionaries inside
        if len(ffhq_data)>0:
            # Load that feature.json has information
            new_ffhq_data[i]['error'] = False
            # Create data key
            new_ffhq_data[i]['data'] = {}
            
            # This block extracts the features that are relevant and binarizes them
            # Begin block
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
            # End block
            
            # Load image in images with features list
            existe_json.append(i)
            
            # Begin test block
            # Makes a list of the possible values ​​of a specific key from feature.json
            # Select key
            clave = ffhq_data[0]['faceAttributes']['headPose']['yaw']
            if not clave in control_clave:
                control_clave.append(clave)
            # End test block
        else:
            # Load that feature.json dont has information
            new_ffhq_data[i]['error'] = True
            # Load image in images without features list
            no_existe_json.append(i)

    # Load images with and without features lists
    new_ffhq_data['feature'] = {'existe':existe_json, 'no existe': no_existe_json}

    # Save reduced and binarized dataset
    print('Generando dataset reducido y binarizado')
    with open('{}.json'.format(dataset_binarizada_output), 'w') as file:
        json.dump(new_ffhq_data, file, indent=4)

    # Show key control
    if verbose:
        print(min(control_clave), max(control_clave))
        
    print('\nFin ejecucion\n')
    return

if __name__ == "__main__":
    main()