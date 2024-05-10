''' 
Facial Landmark Detection in Python with OpenCV

Detection from web cam
'''

# Import Packages
import cv2
import os
import urllib.request as urlreq
import numpy as np
import procesamiento as pr


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
webcam_cap = cv2.VideoCapture(0)

iter_p = 0

while(True):
    # read webcam
    _, frame = webcam_cap.read()

    # convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using the haarcascade classifier on the "grayscale image"
    faces = detector.detectMultiScale(gray)
    
    # save last instance of detected image
    cv2.imwrite('face-detect.jpg', frame) 

    for (x,y,w,d) in faces:
        # Detect landmarks on "gray"
        _, landmarks = landmark_detector.fit(gray, np.array(faces))
        
        lista = landmarks[0]
        print(len(landmarks))
        if 1:
            for i in landmarks:
                centroideder=np.mean(i[0][36:42], axis=0)
                centroideder, centroideizq, unidad, origen_ojo, distojos, distfrente_ojo, distboca_ojo, angulo_cara, angulo_ojo_derecho, angulo_ojo_izquierdo, valores_elipse_ojoder, valores_elipse_ojoizq = pr.calculos(i[0][36:42], i[0][42:48], i[0][48:55], i[0][17:22])
                cv2.circle(frame, (int(centroideder[0]), int(centroideder[1])), 1, (0, 255, 0), 2)
                cv2.circle(frame, (int(centroideizq[0]), int(centroideizq[1])), 1, (0, 255, 0), 2)
                cv2.circle(frame, (int(origen_ojo[0]), int(origen_ojo[1])), 1, (0, 255, 0), 2)
                cv2.circle(frame, (int(centroideder[0]), int(centroideder[1])), 1, (0, 255, 0), 2)
        
        lista_p = landmarks
        if 0:
            if iter_p == 0:
                print(lista_p)
                print(lista_p[0])
                print(lista_p[0][0])
                print(lista_p[0][0][0])
                print(lista_p[0][0][0][0])
                iter_p += 1

        for landmark in landmarks:
            for x,y in landmark[0]:
                # display landmarks on "frame/image,"
                # with blue colour in BGR and thickness 2
                cv2.circle(frame, (int(x), int(y)), 1, (255, 0, 0), 2)

       
    
    # Show image
    cv2.imshow("frame", frame)

    # terminate the capture window
    if cv2.waitKey(20) & 0xFF  == ord('q'):
        webcam_cap.release()
        cv2.destroyAllWindows()
        break
