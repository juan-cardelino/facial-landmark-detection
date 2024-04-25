''' 
Facial Landmark Detection in Python with OpenCV

Detection from web cam
'''

# Import Packages
import cv2
import os
import urllib.request as urlreq
import numpy as np
import json

# location of the models
data_dir = "data"
input_dir = "input"

# save face detection algorithm's url in haarcascade_url variable
haarcascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml"

# save face detection algorithm's name as haarcascade
haarcascade = "haarcascade_frontalface_alt2.xml"
haarcascade_clf = "data/" + haarcascade

# check if data folder is in working directory
if (os.path.isdir('data')):
    # check if haarcascade is in data directory
    if (haarcascade in os.listdir('data')):
        print("File exists")
    else:
        # download file from url and save locally as haarcascade_frontalface_alt2.xml
        urlreq.urlretrieve(haarcascade_url, haarcascade_clf)
        print("File downloaded")
else:
    # create data folder in current directory
    os.mkdir('data')
    # download haarcascade to data folder
    urlreq.urlretrieve(haarcascade_url, haarcascade_clf)
    print("File downloaded")

# create an instance of the Face Detection Cascade Classifier
detector = cv2.CascadeClassifier(haarcascade_clf)

# save facial landmark detection model's url in LBFmodel_url variable
LBFmodel_url = "https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml"

# save facial landmark detection model's name as LBFmodel
LBFmodel = "LFBmodel.yaml"
LBFmodel_file = "data/" + LBFmodel

# check if data folder is in working directory
if (os.path.isdir('data')):
    # check if Landmark detection model is in data directory
    if (LBFmodel in os.listdir('data')):
        print("File exists")
    else:
        # download file from url and save locally as haarcascade_frontalface_alt2.xml
        urlreq.urlretrieve(LBFmodel_url, LBFmodel_file)
        print("File downloaded")
else:
    # create data folder in current directory
    os.mkdir('data')
    # download Landmark detection model to data folder
    urlreq.urlretrieve(LBFmodel_url, LBFmodel_file)
    print("File downloaded")

# create an instance of the Facial landmark Detector with the model
landmark_detector  = cv2.face.createFacemarkLBF()
landmark_detector.loadModel(LBFmodel_file)

# get image from webcam
print ("checking webcam for connection ...")
#webcam_cap = cv2.VideoCapture(1)

#_, frame = webcam_cap.read()
imagen = 3
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


numero_cara = 0
for (x,y,w,d) in faces:
    # Detect landmarks on "gray"
    _, landmarks = landmark_detector.fit(gray, np.array(faces))
    
    lista = np.ones([68, 2])
    numero_cara = numero_cara + 1
    
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
        
        caras = {
            "contorno":contorno.tolist(),
            "ceja derecha":cejader.tolist(),
            "ceja izquierda":cejaizq.tolist(),
            "tabique":tabique.tolist(),
            "fosas nasales":fosas.tolist(),
            "ojo derecho":ojoder.tolist(),
            "ojo izquierdo":ojoizq.tolist(),
            "labio superior":labiosup,
            "labio inferior":labioinf
        }

if 0:
    f = open("Marcadores.txt", "w")
    for x, y in lista:
        f.write(str(x)+" "+str(y)+"\n")
    f.close()
else:
    contorno = lista[0:17]
    cejader = lista[17:22]
    cejaizq = lista[22:27]
    tabique = lista[27:31]
    fosas = lista[31:36]
    ojoder = lista[36:42]
    ojoizq = lista[42:48]
    labiosup = lista[48:55].tolist()+lista[61:64].tolist()
    labioinf = lista[55:61].tolist()+lista[64:68].tolist()
    deteccion = {
        "caras":{
            "contorno":contorno.tolist(),
            "ceja derecha":cejader.tolist(),
            "ceja izquierda":cejaizq.tolist(),
            "tabique":tabique.tolist(),
            "fosas nasales":fosas.tolist(),
            "ojo derecho":ojoder.tolist(),
            "ojo izquierdo":ojoizq.tolist(),
            "labio superior":labiosup,
            "labio inferior":labioinf
        }
    }
    
    with open('deteccion.json', 'w') as file:
        json.dump(deteccion, file, indent=4)

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

