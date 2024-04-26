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

def guardado(a_guardar, verbose = True):
    if verbose:
        with open('deteccion.json', 'w') as file:
            json.dump(a_guardar, file, indent=4)
        print('marcadore guardado en formato json')       
    else:
        f = open("Marcadores.txt", "w")
        for x, y in a_guardar:
            f.write(str(x)+" "+str(y)+"\n")
        f.close()
        print('marcadores guardado en formato txt')
    return

input_dir = "input"
detector = cv2.CascadeClassifier("data/haarcascade_frontalface_alt2.xml")
landmark_detector  = cv2.face.createFacemarkLBF()
landmark_detector.loadModel("data/LFBmodel.yaml")

# get image from webcam
#print ("checking webcam for connection ...")
#webcam_cap = cv2.VideoCapture(1)

#_, frame = webcam_cap.read()
print('cargando archivo')
imagen = 0
if imagen == 0:
    input_fname = os.path.join(input_dir, 'input2.png')
elif imagen == 1:
    input_fname = os.path.join(input_dir, 'juan2.jpg')
elif imagen == 2:
    input_fname = os.path.join(input_dir, 'paisaje.jpg')
elif imagen == 3:
    input_fname = os.path.join(input_dir, 'House.jpg')
frame = cv2.imread(input_fname)

# convert frame to grayscale
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Detect faces using the haarcascade classifier on the "grayscale image"
faces = detector.detectMultiScale(gray)

print("caras detectadas: ",len(faces))
    
#deteccion = {"caras":np.ones(len(faces)-1).tolist()}
deteccion = {"caras":[]}
iter = 0
for (x,y,w,d) in faces:
    deteccion["caras"] = deteccion["caras"]+[1]
    # Detect landmarks on "gray"
    _, landmarks = landmark_detector.fit(gray, np.array(faces))
    
    lista = np.ones([68, 2])
    
    for landmark in landmarks:
        k = 0
        for x,y in landmark[0]:
            k = k + 1
            # display landmarks on "frame/image,"
            # with blue colour in BGR and thickness 2
            #if k>=37 and k<=48:
            if 1:
                cv2.circle(frame, (int(x), int(y)), 1, (255, 0, 0), 5)
                #cv2.putText(frame, str(k) ,(int(x)+10, int(y)), cv2.FONT_HERSHEY_SIMPLEX , 1, (255,0,0), 1)
            lista[k-1][0] = x
            lista[k-1][1] = y
    deteccion["caras"][iter] = {
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
    guardado(deteccion)
    

    # save last instance of detected image
    cv2.imwrite('face-detect.jpg', frame)    

    # Show image
    cv2.imshow("frame", cv2.resize(frame,(1000,800)))
    cv2.waitKey()
    # terminate the capture window
    #if cv2.waitKey(20) & 0xFF  == ord('q'):
    #    webcam_cap.release()
    #    cv2.destroyAllWindows()
    #    break

    cv2.destroyAllWindows()
else:
    print('Error: No se detecto cara')
print('Ejecucion finalizada')

