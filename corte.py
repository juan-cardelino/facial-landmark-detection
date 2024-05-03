import json
import cv2

def cuerpo(imagenes, verbose = 1, input_dir = 'detected'):
        for imagen in imagenes:
                if verbose>=1:
                        nombre_j = imagen[:imagen.rfind('.')]
                        with open('Json/'+nombre_j + "_data.json") as archivo:
                                deteccion = json.load(archivo)
                        
                        boundingbox = deteccion["boundingbox"]

                        # Reading the image 
                        image = cv2.imread(input_dir+'/'+imagen)
  
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

                        start_x = boundingbox[0]
                        end_x = boundingbox[0]+boundingbox[2]
                        start_y = boundingbox[1]
                        end_y = boundingbox[1]+boundingbox[3]

                        image_cropped = rotated_image[start_y:end_y, start_x:end_x]
                
                if verbose >=2:

                        cv2.imshow("face rotated:", cv2.resize(image_cropped,(900,800)))
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                
        
        return

cuerpo(['input2.jpg'], 2)
