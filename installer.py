import os
import urllib.request as urlreq
import json

def check_folder(name):
    # Check folder exists
    if (os.path.isdir(name)):
        print("Folder {} already exists".format(name))
    else:
        # If not, create folder
        os.mkdir(name)
        print("Folder {} created".format(name))
    return

with open('configuracion.json') as file:
    configuracion = json.load(file)

print("Downloading models")

# location of the models
data_dir = configuracion['path']['model_dir']
    
# save face detection algorithm's url in haarcascade_url variable
haarcascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml"

# save face detection algorithm's name as haarcascade
haarcascade = "haarcascade_frontalface_alt2.xml"
haarcascade_clf = data_dir+"/" + haarcascade

# check if data folder is in working directory
if (os.path.isdir(data_dir)):
    # check if haarcascade is in data directory
    print("File {} already exists".format(data_dir))
    if (haarcascade in os.listdir(data_dir)):
        print("Face detection model already exists")
    else:
        # download file from url and save locally as haarcascade_frontalface_alt2.xml
        urlreq.urlretrieve(haarcascade_url, haarcascade_clf)
        print("Face detection model downloaded")
else:
    # create data folder in current directory
    os.mkdir(data_dir)
    print("Folder {} created".format(data_dir))
    # download haarcascade to data folder
    urlreq.urlretrieve(haarcascade_url, haarcascade_clf)
    print("Face detection model downloaded")

# save facial landmark detection model's url in LBFmodel_url variable
LBFmodel_url = "https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml"

# save facial landmark detection model's name as LBFmodel
LBFmodel = "LFBmodel.yaml"
LBFmodel_file = data_dir+"/" + LBFmodel

# check if data folder is in working directory
if (os.path.isdir(data_dir)):
    # check if Landmark detection model is in data directory
    if (LBFmodel in os.listdir(data_dir)):
        print("Landmark detection model already exists")
    else:
        # download file from url and save locally as haarcascade_frontalface_alt2.xml
        urlreq.urlretrieve(LBFmodel_url, LBFmodel_file)
        print("Landmark detection model downloaded")
else:
    # create data folder in current directory
    os.mkdir(data_dir)
    # download Landmark detection model to data folder
    urlreq.urlretrieve(LBFmodel_url, LBFmodel_file)
    print("Landmark detection model downloaded")

print("All models ready \n")

print("Creating folders")

check_folder(configuracion['path']['input_dir'])
check_folder(configuracion['path']['output_dir'])
check_folder(configuracion['path']['detect_dir'])
check_folder(configuracion['path']['aligned_dir'])
check_folder(configuracion['path']['json_dir'])

print("All folders ready")