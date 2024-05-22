# Facial feature detection

This program use a facial detection algorithm to detect faces and its landmarks on an image. The landmakrs are use to calculate facial features.

### Instalation

In order to use this code is necesary to install the modules in requirements.txt and run the installer.py program to download the models use.

Necesary instalation to run the code:
    `pip install requirements.txt`
    `python.exe installer.py`

### Face Detection

This detects faces and facial landmarks on an image, the image has to be located in local directory: input.

A python file to detect facial landmarks via webcam.(Dudas)
A jupyter notebook to detect image files located in directory.(Dudas)

This project uses Python 3.10, openCV 4.9.0.80, numpy 1.26.4 and scipy 1.13.0

The model used for landmark detection can be got from: 
https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml

Also the model used for face detection can be found here:
https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml

