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
    configuration = json.load(file)

print('Start execution')
print("Downloading models")

# location of the models
models_dir = configuration['path']['model_dir']

# save face detection algorithm's url in haarcascade_url variable
haarcascade_url = configuration['general']['face detection model_url']

# save face detection algorithm's name as haarcascade
haarcascade = configuration['general']['face detection model']
haarcascade_clf = '{}/{}'.format(models_dir, haarcascade)

# check if data folder is in working directory
if (os.path.isdir(models_dir)):
    # check if haarcascade is in data directory
    print("File {} already exists".format(models_dir))
    if (haarcascade in os.listdir(models_dir)):
        print("Face detection model already exists")
    else:
        # download file from url and save locally as haarcascade_frontalface_alt2.xml
        urlreq.urlretrieve(haarcascade_url, haarcascade_clf)
        print("Face detection model downloaded")
else:
    # create data folder in current directory
    os.mkdir(models_dir)
    print("Folder {} created".format(models_dir))
    # download haarcascade to data folder
    urlreq.urlretrieve(haarcascade_url, haarcascade_clf)
    print("Face detection model downloaded")

# save facial landmark detection model's url in LBFmodel_url variable
LBFmodel_url = configuration['general']['landmark detection model_url']

# save facial landmark detection model's name as LBFmodel
LBFmodel = configuration['general']['landmark detection model']
LBFmodel_file = '{}/{}'.format(models_dir, LBFmodel)

# check if data folder is in working directory
if (os.path.isdir(models_dir)):
    # check if Landmark detection model is in data directory
    if (LBFmodel in os.listdir(models_dir)):
        print("Landmark detection model already exists")
    else:
        # download file from url and save locally as haarcascade_frontalface_alt2.xml
        urlreq.urlretrieve(LBFmodel_url, LBFmodel_file)
        print("Landmark detection model downloaded")
else:
    # create data folder in current directory
    os.mkdir(models_dir)
    # download Landmark detection model to data folder
    urlreq.urlretrieve(LBFmodel_url, LBFmodel_file)
    print("Landmark detection model downloaded")

print("All models ready \n")

print("Creating folders")

for i in configuration['path']:
    check_folder(configuration['path'][i])
print("All folders ready")

print('End execution')