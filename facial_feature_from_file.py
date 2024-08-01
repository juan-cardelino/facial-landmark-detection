'''
This program calculate facial features from images in an input folder and save them in a json file

'''

import os
import json
import file_landmark as fland
import procesamiento as pr
import alinear
import cv2

def main():
    '''
    Run program:
    
    It starts by loading initial setup from configuracion.json

    The program has 3 stages.

    The first stage read all the images in the input folder, creates a list of them and use the find_landmark module to find if there are faces in each image. If there are faces in an image, the corresponding landmarks are calculated and saved.

    The second stage use the previously found landmarks and the procesamiento module to calculate all facial features and save them.

    The third stage use the alinear module to rotate the image to line up the biggest face and cropp ir by the bounding box.
    '''
    # Initial setup
    with open('configuracion.json') as file:
        configuration = json.load(file)

    raw_input = configuration["path"]["input_dir"]
    detected_output = configuration["path"]["detect_dir"]
    aligned_output = configuration["path"]["aligned_dir"]
    json_dir = configuration["path"]["json_dir"]
    verbose = configuration["pipeline"]["from_file"]["verbose"]
    minimum_face_width = configuration["pipeline"]["from_file"]["minimo_ancho_de_cara"]
    stage = configuration["pipeline"]["from_file"]["etapas"]
    json_suffix_detect = configuration['general']['json_suffix_detect']
    json_suffix_data = configuration['general']['json_suffix_data']
    resize = configuration["general"]["resize"]


    file = os.listdir(raw_input)

    print("Se corren {} de 3 etapas".format(stage))

    # Stage 1, get facial landmarks
    if stage > 0:
        print("\nInicio etapa 1")
        fland.find_landmarks(file, minimum_face_width, verbose, raw_input, detected_output, json_dir, json_suffix_detect, resize)
        print("Fin etapa 1\n")
        
    # Stage 2, calculate facial feature from landmarks
    if stage > 1:
        print("Inicio etapa 2")
        images = []
        l_suffix = len(json_suffix_detect)+5
        for i in os.listdir(json_dir):
            # Detect json suffix
            if i[-l_suffix:] == json_suffix_detect+'.json':
                images.append(i[:-l_suffix-1])
        max_faces = 1

        for image in images:
            # Get landmark from json
            image_file, right_eye, left_eye, forehead, mouth, boundingbox, face_amount = pr.load_landmarks(image, max_faces, json_dir)

            for i in range(face_amount):
                # Calculate facial features
                right_eye_centroid, left_eye_centroid, unit, eyes_origin, eyes_distance, eyes_forehead_distance, eyes_mouth_distance, face_angle, right_eye_angle, left_eye_angle, right_eye_ellipse_values, left_eye_ellipse_values = pr.calculate_facial_feature(right_eye[i], left_eye[i], forehead[i], mouth[i])

                # Structured storage
                if i == 0:
                    pr.save_features(image_file, right_eye_centroid, left_eye_centroid, unit, eyes_origin, eyes_distance, eyes_forehead_distance, eyes_mouth_distance, face_angle, right_eye_angle, left_eye_angle, right_eye_ellipse_values, left_eye_ellipse_values, boundingbox[0], image, json_dir, json_suffix_data)
                    print('mejor cara guardada en {}_{}.json'.format(image, json_suffix_data))

        print("Fin etapa 2\n")

    # Stage 3, 
    if stage > 2:
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
            image_file, boundingbox, angle = alinear.get_json_data(data, json_dir, json_suffix_data)
            print(image_file)
            # Get frame
            frame = cv2.imread('{}/{}'.format(raw_input, image_file))
            # Rotate frame using boundingbox
            frame_rotated = alinear.rotate(frame, boundingbox, -angle)
            # Cropp frame using boundingbox
            frame_cropped = alinear.cropp(frame_rotated, boundingbox)
            # Save frame in aligned folder
            cv2.imwrite('{}/{}.jpg'.format(aligned_output, data), frame_cropped)
            print('{}.jpg guardada en {} folder'.format(data, aligned_output))
        print("\nFin etapa 3\n")

    print("Fin ejecucion")
    return

if __name__ == "__main__":
    main()