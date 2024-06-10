import cv2
import numpy as np
import json

# Calcular color mediante promedios de numpy
def numpy_color(frame):
    aux = np.mean(np.mean(frame, axis=0), axis=0)
    output = frame[0][0]
    output[0] = aux[0]
    output[1] = aux[1]
    output[2] = aux[2]
    return output

# Calcuar color mediante resize de cv2
def cv2_color(frame):
    return cv2.resize(frame,(1,1))[0][0]

# Cortar imagen en base a boundingnbox, porcentaje achica el recorte
def cropp(frame, boundingbox, porcentaje = 0.2):
    offset_x = int(boundingbox[1][0]-boundingbox[0][0])
    offset_y = int(boundingbox[1][1]-boundingbox[0][1])
    
    start_x = boundingbox[0][0]+ int(offset_x*porcentaje)
    end_x = boundingbox[1][0]- int(offset_x*porcentaje)
    start_y = boundingbox[0][1]+ int(offset_y*porcentaje)
    end_y = boundingbox[1][1]- int(offset_y*porcentaje)
    return frame[start_y:end_y, start_x:end_x]

folder_path = 'Pruebas/Fotos/'
foto = '00055.png'

frame = cv2.imread(folder_path+foto)

# Boundigbox especifico de la imagen, si se usa otra imagen se podria calcular con funciones de file_landmark
boundingbox = [[224,384],[773,946]]

frame_cropped = cropp(frame, boundingbox)

color_cara_numpy = numpy_color(frame_cropped)
print('color numpy: {}'.format(color_cara_numpy))
color_cara_cv2 = np.array(cv2_color(frame_cropped))
print('color cv2: {}'.format(color_cara_cv2))

cv2.imshow("cara", frame_cropped)
cv2.waitKey(0)
cv2.imshow("color numpy", cv2.resize(np.array([[color_cara_numpy]]), (200, 200)))
cv2.waitKey(0)
cv2.imshow("color cv2", cv2.resize(np.array([[color_cara_cv2]]), (200, 200)))
cv2.waitKey(0)
cv2.destroyAllWindows()
        

