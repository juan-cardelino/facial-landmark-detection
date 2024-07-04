''' 
Facial Landmark Detection in Python with OpenCV

Detection from web cam
'''

# Import Packages
import cv2
import os
import numpy as np
import json
import graficar

def save_in_json(to_save, name, json_dir = "Json", json_suffix = 'deteccion'):
    # Create and write json file
    with open('{}/{}_{}.json'.format(json_dir, name, json_suffix), 'w') as file:
        json.dump(to_save, file, indent=4)
    
    if len(to_save['caras']) != 0:
        print('Marcadores guardados en formato json')       
    return

def find_landmarks(images, minimum_face_width = 100, verbose = 1, input_dir="input", output_dir="detected", json_dir="Json", json_suffix = 'deteccion', resize = (1920,1080)):
    
    # Initial setup
    with open('configuracion.json') as file:
        configuration = json.load(file)
    
    models_dir = configuration['path']['model_dir']
    haarcascade = configuration['general']['face detection model']
    LBFmodel = configuration['general']['landmark detection model']
    
    # Load models
    detector = cv2.CascadeClassifier("{}/{}".format(models_dir, haarcascade))
    landmark_detector  = cv2.face.createFacemarkLBF()
    landmark_detector.loadModel("{}/{}".format(models_dir, LBFmodel))
    print('modelos cargados')
    
    for image in images: 
        
        # Remove image extension
        json_name = image[:image.rfind('.')]
        
        # Show image being processed
        print("\narchivo: {}".format(json_name))
        
        # Open image
        input_frame = os.path.join(input_dir, str(image))
        frame = cv2.imread(input_frame)

        # Get minimum detection 
        minimum_detection_width = max([minimum_face_width, frame.shape[:2][1]//16])
    
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces using the haarcascade classifier on the "grayscale frame"
        faces = detector.detectMultiScale(gray)
        
        print("caras detectadas: {}".format(len(faces)))

        # Initialize detection dict
        detection = {"image file":image, "caras":[], "cantidad de caras":0, "Error":"No se encontraron errores"}
        
        for (x,y,w,d) in faces:
            # Create boundingbox
            boundingbox = [int(x), int(y), int(w), int(d)]
            if verbose >= 5:
                # Show boundingbox
                frame = graficar.boundingbox(frame, boundingbox, (255,255,255))
            # Control if boundingdox width is bigger than minimum detection width
            if w>minimum_detection_width:
                # Detect landmarks on "gray"
                _, auxiliar_landmarks = landmark_detector.fit(gray, np.array([[x, y, w, d]]))
                landmarks = auxiliar_landmarks[0][0]
    
                if verbose >= 2:
                    # Show landmarks
                    frame = graficar.graph_circle(frame, landmarks, (255, 0, 0), int(w/64))
                
                # Load detection dict
                detection["cantidad de caras"] += 1
                detection["caras"].append({
                    "boundingbox":boundingbox,
                    "contorno":landmarks[0:17].tolist(),
                    "ceja derecha":landmarks[17:22].tolist(),
                    "ceja izquierda":landmarks[22:27].tolist(),
                    "tabique":landmarks[27:31].tolist(),
                    "fosas nasales":landmarks[31:36].tolist(),
                    "ojo derecho":landmarks[36:42].tolist(),
                    "ojo izquierdo":landmarks[42:48].tolist(),
                    "labio superior":landmarks[48:55].tolist()+landmarks[61:64].tolist(),
                    "labio inferior":landmarks[55:61].tolist()+landmarks[64:68].tolist()
                })

        # Control if face was detected
        if len(faces) != 0:
            # Show amount of faces detected
            print("cantidad de caras suficientemente grandes: {}".format(str(len(detection["caras"]))))
            # Control if there are big enough faces
            if len(detection["caras"])==0:
                # Save error: small faces
                detection["Error"]="Mejor cara demasiado chica"
                # Show error
                print("Error: Mejor cara demasiado chica")
            else:
                if verbose > 4:
                    # Save processed image in output folder
                    cv2.imwrite('{}/{}.jpg'.format(output_dir, json_name), frame)
                    # Show image saved
                    print("{}.jpg guardado en {}".format(json_name ,output_dir))

            if verbose >= 3:
                # Show image on screen
                cv2.imshow("frame", cv2.resize(frame, resize))
                cv2.waitKey()
                cv2.destroyAllWindows()
        else:
            # Save error: face not detected
            detection["Error"] = "No se detecto ninguna cara"
            # Show error
            print('Error: No se detecto cara')
        # Sort faces detected by boundingbox width
        detection['caras'] = sorted(detection['caras'], key=lambda aux:aux['boundingbox'][2], reverse=True)
        # Save dict in json
        save_in_json(detection, json_name, json_dir=json_dir, json_suffix=json_suffix)
    return