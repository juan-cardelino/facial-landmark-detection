import os
import urllib.request as urlreq

print("Downloading models")

# location of the models
data_dir = "data"
    
# save face detection algorithm's url in haarcascade_url variable
haarcascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml"

# save face detection algorithm's name as haarcascade
haarcascade = "haarcascade_frontalface_alt2.xml"
haarcascade_clf = "data/" + haarcascade

# check if data folder is in working directory
if (os.path.isdir('data')):
    # check if haarcascade is in data directory
    print("File data already exists")
    if (haarcascade in os.listdir('data')):
        print("Face detection model already exists")
    else:
        # download file from url and save locally as haarcascade_frontalface_alt2.xml
        urlreq.urlretrieve(haarcascade_url, haarcascade_clf)
        print("Face detection model downloaded")
else:
    # create data folder in current directory
    os.mkdir('data')
    print("File data already exists")
    # download haarcascade to data folder
    urlreq.urlretrieve(haarcascade_url, haarcascade_clf)
    print("Face detection model downloaded")

# create an instance of the Face Detection Cascade Classifier
#detector = cv2.CascadeClassifier(haarcascade_clf)

# save facial landmark detection model's url in LBFmodel_url variable
LBFmodel_url = "https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml"

# save facial landmark detection model's name as LBFmodel
LBFmodel = "LFBmodel.yaml"
LBFmodel_file = "data/" + LBFmodel

# check if data folder is in working directory
if (os.path.isdir('data')):
    # check if Landmark detection model is in data directory
    if (LBFmodel in os.listdir('data')):
        print("Landmark detection model already exists")
    else:
        # download file from url and save locally as haarcascade_frontalface_alt2.xml
        urlreq.urlretrieve(LBFmodel_url, LBFmodel_file)
        print("Landmark detection model downloaded")
else:
    # create data folder in current directory
    os.mkdir('data')
    # download Landmark detection model to data folder
    urlreq.urlretrieve(LBFmodel_url, LBFmodel_file)
    print("Landmark detection model downloaded")

print("All models ready \n")

print("Creating folders")

if (os.path.isdir('input')):
    print("Folder input already exists")
else:
    os.mkdir('input')
    print("Folder input created")

if (os.path.isdir('output')):
    print("Folder out already exists")
else:
    os.mkdir('output')
    print("Folder output created")
        
if (os.path.isdir('detected')):
    print("Folder detected already exists")
else:
    os.mkdir('detected')
    print("Folder detected created")

print("All folders ready")