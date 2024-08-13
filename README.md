# Facial feature detection

This program calculates facial features from an image. It first detects faces and facial landmarks on a given image.

<p align="center">
    <img src=Readme_images/Read1.jpg width = "75%">
</p>

Then uses the landmarks to perform the necessary calculations to obtain facial features.

The calculation method can be found in [miro/Calculos: Facial Feature][miro]

## Index

### Readme
[Installation](#installation)

[From file tutorial](#tutorial-facial-feature-from-file)

[From video tutorial](#tutorial-facial-feature-from-video)

[Examples](#examples)

[Credits](#credits)

### [Wiki](https://github.com/juan-cardelino/facial-landmark-detection/wiki)

[Programs](https://github.com/juan-cardelino/facial-landmark-detection/wiki/Programs)

[Modules](https://github.com/juan-cardelino/facial-landmark-detection/wiki/Modules)

[Database](https://github.com/juan-cardelino/facial-landmark-detection/wiki/Database)

[Configuration](https://github.com/juan-cardelino/facial-landmark-detection/wiki/Configuration)

## Installation

In order to use this code is necessary to install the modules in requirements.txt and run the installer.py program to download the models use.

Necessary installation to run the code:

    pip install -r requirements.txt
    
    python.exe installer.py

This project uses Python 3.10, openCV 4.9.0.80, numpy 1.26, scipy 1.13.0, json and math

To verify that the models were downloaded, the files haarcascade_frontalface_alt2.xml and LFBmodel.yaml must be in the folder data

The model used for face detection can be got from:
https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_alt2.xml

The model used for landmark detection can be got from: 
https://github.com/kurnianggoro/GSOC2017/raw/master/data/lbfmodel.yaml

The model used for approximating point to an elipse can be found here:
https://github.com/cjgb/ellipses/blob/dev/mylib.py

## Tutorial facial feature from file

To process image files, the following process must be followed:

### Preprocess

Insert image/es to process in Input folder

### Run code

    python.exe facial_feature_from_file.py

### Results

This code has 3 stages, the amount of stages to perform can be specified in the [configuracion](https://github.com/juan-cardelino/facial-landmark-detection/wiki/Configuration#from_file).json file

#### Stage 1

For each image in Input folder, a json file is created in Json folder with the format image_name_detection.json. If the image has faces with a boundingbox bigger than 100 pixels, the json is load with the landmarks of all big enough faces, if not, the json is load with an error message (face not detected or face not big enough).

#### Stage 2

Landmarks found in previous stage are use to perform the features calculations (found in [miro/Calculos: Facial Feature][miro]). Facial features from the biggest face are save in Json folder with the format image_name_data.json.

#### Stage 3

It creates a copy of the images with faces, each images is rotated by the angle of the biggest face to line it up, then the image is cropp by the boundingbox of the same face. All new images are save in Aligned folder.

## Tutorial facial feature from video

### Preprocess

Configurate the video input to process in [configuracion](https://github.com/juan-cardelino/facial-landmark-detection/wiki/Configuration#from_video).json file (if video_file = 0, the video will be taken from camera).

### Run code

    python.exe facial_feature_from_video.py

### Results

This program cut the video in frames, for each frame, it finds the facial landmarks and calculate the facial features, then graph all the landmarks and the ellipse of the eyes. It opens a new window where the video is shown with the graphed frames. The graphed frames can be save in video and/or image format, depending on [configuration](https://github.com/juan-cardelino/facial-landmark-detection/wiki/Configuration#from_video).

## Examples

### Eyes centroids
<p align="center">
    <img src=Readme_images/Read2.jpg width = "50%">
</p>

### Eyes-forehead distance
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

[miro]:https://miro.com/app/board/uXjVKVwTq8w=/

# More information

Find more information on how to execute the programs in [pipeline](https://docs.google.com/document/d/13zKQjNVyC3yyjy1iC3ybzMHDT74Gb_B5hFoXsu0esgc/edit#heading=h.wj7nhmmfcxck)

Find more information about feature calculation on [miro][miro]

Find more tecnical information in [documentation][https://hisokam324.github.io/]

