''' 
Facial Landmark Detection in Python with OpenCV

Detection from web cam
'''

# Import Packages
import cv2
import os
import numpy as np
import json

def guardado(a_guardar, nombre, json_dir = "Json"):
    with open(json_dir+'/'+nombre+'_deteccion.json', 'w') as file:
        json.dump(a_guardar, file, indent=4)
    print('Marcadores guardados en formato json')       
    return

def find_landmarks(imagenes, minimo_ancho_de_cara = 100, verbose = 1, input_dir="input", output_dir="detected", json_dir="Json",model_dir = "data"):
    
    detector = cv2.CascadeClassifier(model_dir+"/haarcascade_frontalface_alt2.xml")
    landmark_detector  = cv2.face.createFacemarkLBF()
    landmark_detector.loadModel(model_dir+"/LFBmodel.yaml")
    print('modelos cargados')
    
    for imagen in imagenes: 
        
        nombre_j = imagen[:imagen.rfind('.')]
        
        #print("archivo: "+nombre_j)
        
        input_fname = os.path.join(input_dir, str(imagen))
        frame = cv2.imread(input_fname)
    
        minimo_ancho_de_cara = max([minimo_ancho_de_cara, frame.shape[:2][1]//16])
    
        # convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces using the haarcascade classifier on the "grayscale image"
        faces = detector.detectMultiScale(gray)
        
        print("caras detectadas: ",len(faces))

        deteccion = {"caras":[], "cantidad de caras":0, "Error":"No se encontraron errores"}
        iter = 0
        for (x,y,w,d) in faces:
            if verbose >= 4:
                cv2.rectangle(frame, (x, y), (x+w, y+d), (255,255,255), int(w/64))
            if w>minimo_ancho_de_cara:
                # Detect landmarks on "gray"
                _, landmarks = landmark_detector.fit(gray, np.array([[x, y, w, d]]))
    
                lista = landmarks[0][0]
    
                if verbose >= 2:
                    #cv2.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+w)), (255,255,255), int(w/64))
                    #cv2.circle(frame, (int(x+w/2), int(y+w/2)), int(w/np.sqrt(2)), (255, 0, 0), int(w/64))
                    for xx, yy in lista:
                        cv2.circle(frame, (int(xx), int(yy)), 1, (255, 0, 0), int(w/64))
                        #cv2.putText(frame, str(k) ,(int(x)+10, int(y)), cv2.FONT_HERSHEY_SIMPLEX , 1, (255,0,0), 1)           
   
                deteccion["caras"] = deteccion["caras"]+[1]
                deteccion["cantidad de caras"] = deteccion["cantidad de caras"]+1
                deteccion["caras"][iter] = {
                    "boundingbox":[int(x), int(y), int(w), int(d)],
                    "contorno":lista[0:17].tolist(),
                    "ceja derecha":lista[17:22].tolist(),
                    "ceja izquierda":lista[22:27].tolist(),
                    "tabique":lista[27:31].tolist(),
                    "fosas nasales":lista[31:36].tolist(),
                    "ojo derecho":lista[36:42].tolist(),
                    "ojo izquierdo":lista[42:48].tolist(),
                    "labio superior":lista[48:55].tolist()+lista[61:64].tolist(),
                    "labio inferior":lista[55:61].tolist()+lista[64:68].tolist()
                }
                iter = iter + 1
  
        if len(faces) != 0:
            print("cantidad de caras suficientemente grandes: "+str(len(deteccion["caras"])))
            if len(deteccion["caras"])==0:
                deteccion["Error"]="Mejor cara demasiado chica"
                print("Error: Mejor cara demasiado chica")
            else:
                cv2.imwrite(output_dir+'/'+nombre_j+'.jpg', frame)
                print(nombre_j+".jpg guardado en "+output_dir)
        
            if verbose >= 3:
                cv2.imshow("frame", cv2.resize(frame,(1000,800)))
                cv2.waitKey()
                cv2.destroyAllWindows()
        else:
            deteccion["Error"] = "No se detecto ninguna cara"
            print('Error: No se detecto cara')
        deteccion['caras'] = sorted(deteccion['caras'], key=lambda aux:aux['boundingbox'][2], reverse=True)
        guardado(deteccion, nombre_j, json_dir=json_dir)
        #print("")
    #print('Extraccion de landmarks finalizada')
    
    return