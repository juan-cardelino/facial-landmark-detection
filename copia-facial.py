''' 
Facial Landmark Detection in Python with OpenCV

Detection from web cam
'''

# Import Packages
import cv2
import os
import urllib.request as urlreq
import numpy as np

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
#input_fname = os.path.join(input_dir, 'input2.png')
input_fname = os.path.join(input_dir, 'juan2.jpg')
frame = cv2.imread(input_fname)

# convert frame to grayscale
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Detect faces using the haarcascade classifier on the "grayscale image"
faces = detector.detectMultiScale(gray)


for (x,y,w,d) in faces:
    # Detect landmarks on "gray"
    _, landmarks = landmark_detector.fit(gray, np.array(faces))
    
    lista = []
    lista2 = np.ones([68, 2])
    
    for landmark in landmarks:
        k = 0
        for x,y in landmark[0]:
            k = k + 1
            # display landmarks on "frame/image,"
            # with blue colour in BGR and thickness 2
            cv2.circle(frame, (int(x), int(y)), 1, (255, 0, 0), 5)
            cv2.putText(frame, str(k) ,(int(x)+10, int(y)), cv2.FONT_HERSHEY_SIMPLEX , 1, (255,0,0), 1)
            lista.append(str(x)+" "+str(y))
            lista2[k-1][0] = x
            lista2[k-1][1] = y

ojoder = lista2[36:42]
ojoizq = lista2[42:48]

centroideder = np.mean(ojoder, axis= 0)
centroideizq = np.mean(ojoizq, axis= 0)

for x, y in ojoder:
    cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), 5)
for x, y in ojoizq:
    cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), 5)

cv2.circle(frame, (int(centroideder[0]), int(centroideder[1])), 1, (0, 0, 255), 5)
cv2.circle(frame, (int(centroideizq[0]), int(centroideizq[1])), 1, (0, 0, 255), 5)
  
f = open("Marcadores.txt", "w")
for i in lista:
    f.write(str(i)+"\n")
f.close()

# save last instance of detected image
cv2.imwrite('face-detect.jpg', frame)    

# Show image
frame1 = cv2.resize(frame,(600,500))
cv2.imshow("frame", frame1)
cv2.waitKey()
# terminate the capture window
#if cv2.waitKey(20) & 0xFF  == ord('q'):
#    webcam_cap.release()
#    cv2.destroyAllWindows()
#    break

cv2.destroyAllWindows()

