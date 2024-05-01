import json
import cv2

nombre = "data"
with open(nombre + ".json") as archivo:
        deteccion = json.load(archivo)
image = cv2.imread("detected/face-detect.jpg")
boundingbox = deteccion["boundingbox"]

# Reading the image 
image = cv2.imread('detected/face-detect.jpg') 
  
# Extracting height and width from  
# image shape 
height, width = image.shape[:2] 
  
# get the center coordinates of the 
# image to create the 2D rotation 
# matrix 
center = (boundingbox[0]+boundingbox[2]//2,boundingbox[1]+boundingbox[3]//2) 

angulo = deteccion["angulos"]["cara"]
  
# using cv2.getRotationMatrix2D()  
# to get the rotation matrix 
rotate_matrix = cv2.getRotationMatrix2D(center, angulo, 1)
  
# rotate the image using cv2.warpAffine  
# 90 degree anticlockwise 
rotated_image = cv2.warpAffine(image, rotate_matrix, (width, height)) 
  
cv2.imshow("rotated image:", cv2.resize(rotated_image,(900,800)))
cv2.waitKey(0)
cv2.destroyAllWindows()
