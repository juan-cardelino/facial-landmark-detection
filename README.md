# Facial feature detection

This prorgam calculates facial features from an image. It fist detects faces and facial landmarks on a given image.

<p align="center">
    <img src=Readme_images/Read1.jpg width = "75%">
</p>

Then uses the landmarks to perform the necesary calculations to obtain facial features.

The calculation method can by found in [miro/Documentation](https://miro.com/app/board/uXjVKVwTq8w=/)

## Index

[Instalation](https://github.com/juan-cardelino/facial-landmark-detection/blob/master/README.md#instalation)

[Tutorial](https://github.com/juan-cardelino/facial-landmark-detection/blob/master/README.md#tutorial)

[Examples](#examples)

[Credits](#credits)

## Instalation

In order to use this code is necesary to install the modules in requirements.txt and run the installer.py program to download the models use.

Necesary instalation to run the code:

    pip install requirements.txt
    
    python.exe installer.py

This project uses Python 3.10, openCV 4.9.0.80, numpy 1.26, scipy 1.13.0, json and math

To verify that the models were downloaded, the files haarcascade_frontalface_alt2.xml and LFBmodel.yaml must be in the folder data

The model used for face detection can be found here:
https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml

Also the model used for landmark detection can be got from: 
https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml

The model used for aproximating point to an elipse can be got from:
https://github.com/cjgb/ellipses/blob/dev/mylib.py

## Tutorial

To process an image, the following process must be followed:

### Preprocess

Insert image to process in Input folder

### Run code

    python.exe facial-feature-from-file.py

### Results

#### Stage 1

If the image has faces with a boundingbox bigger than 100 pixels, a copy of the image in format jpg is made in detected folder, also a json file is made in the Json folder with the format image_name_deteccion.json with the landmarks off all big enough faces.

#### Stage 2

The landmarks found in the previous stage are use to perfomr the fetures calculations. Facial features are save in Json folder with the format image_name_data.json.



## Examples

### Eyes centroids
<p align="center">
    <img src=Readme_images/Read2.jpg width = "50%">
</p>

### Eyes-forhead distance
<p align="center">
    <img src=Readme_images/Read3.jpg width = "50%">
</p>

### Eyes-mouth distance
<p align="center">
    <img src=Readme_images/Read4.jpg width = "50%">
</p>

## Credits
Landmark detection:
https://github.com/Danotsonof/facial-landmark-detection

elliptic least squares:
https://github.com/cjgb/ellipses


