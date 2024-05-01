''' 
Facial Landmark Detection in Python with OpenCV

Detection from web cam
'''

# Import Packages
import cv2
import os
#import urllib.request as urlreq
import numpy as np
import json

def guardado(a_guardar):
    with open('deteccion.json', 'w') as file:
        json.dump(a_guardar, file, indent=4)
    print('Marcadores guardados en formato json')       
    return

verbose = 4
imagen = 3
minimo_ancho_de_cara = 0

input_dir = "input"
detector = cv2.CascadeClassifier("data/haarcascade_frontalface_alt2.xml")
landmark_detector  = cv2.face.createFacemarkLBF()
landmark_detector.loadModel("data/LFBmodel.yaml")

# get image from webcam
#print ("checking webcam for connection ...")
#webcam_cap = cv2.VideoCapture(1)

#_, frame = webcam_cap.read()
print('cargando archivo')
input_fname = os.path.join(input_dir, str(os.listdir("input")[imagen]))
frame = cv2.imread(input_fname)

#print(os.listdir("input")[0])

# convert frame to grayscale
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Detect faces using the haarcascade classifier on the "grayscale image"
faces = detector.detectMultiScale(gray)

print("caras detectadas: ",len(faces))
    
#deteccion = {"caras":np.ones(len(faces)-1).tolist()}

deteccion = {"caras":[], "cantidad de caras":0, "mejor cara":{}, "Error":"No se encontraron errores"}
iter = 0
mejor_cara = [0, 0]
for (x,y,w,d) in faces:
    if verbose >= 4:
        cv2.rectangle(frame, (x, y), (x+w, y+d), (255,255,255), int(w/64))
    if w>minimo_ancho_de_cara:
        # Detect landmarks on "gray"
        _, landmarks = landmark_detector.fit(gray, np.array([[x, y, w, d]]))
    
        lista = landmarks[0][0]
    
        if verbose >= 2:
            for x, y in lista:
                cv2.circle(frame, (int(x), int(y)), 1, (255, 0, 0), int(w/64))
                #cv2.putText(frame, str(k) ,(int(x)+10, int(y)), cv2.FONT_HERSHEY_SIMPLEX , 1, (255,0,0), 1)           
   
        deteccion["caras"] = deteccion["caras"]+[1]
        deteccion["caras"][iter] = {
            "ancho":int(w),
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
        if mejor_cara[1]<w:
            mejor_cara = [iter, w]
        iter = iter + 1

        

  
if len(faces) != 0:
    deteccion["mejor cara"]["indice"]=int(mejor_cara[0])
    deteccion["mejor cara"]["ancho"]=int(mejor_cara[1])
    if mejor_cara[1]<minimo_ancho_de_cara:
        deteccion["Error"]="Mejor cara demasiado chica"
    guardado(deteccion)
    
    # save last instance of detected image
    cv2.imwrite('detected/face-detect.jpg', frame)    
    if verbose >= 3:
        # Show image
        #cv2.imshow("algo", gray)
        cv2.imshow("frame", cv2.resize(frame,(1000,800)))
        cv2.waitKey()
        # terminate the capture window
        #if cv2.waitKey(20) & 0xFF  == ord('q'):
        #    webcam_cap.release()
        #    cv2.destroyAllWindows()
        #    break

        cv2.destroyAllWindows()
else:
    deteccion["Error"] = "No se detecto ninguna cara"
    guardado(deteccion)
    print('Error: No se detecto cara')
print('Ejecucion finalizada')

