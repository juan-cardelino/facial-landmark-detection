import cv2
import numpy as np
import json

# Calcular color mediante promedios de numpy
def numpy_color(frame):
    promedio = np.mean(np.mean(frame, axis=0), axis=0)
    # Aca se usa un pixel del frame, porque si se devuelve un arreglo normal openCv hace problemas con punteros
    output = frame[0][0]
    output[0] = promedio[0]
    output[1] = promedio[1]
    output[2] = promedio[2]
    return output

# Calcuar color mediante resize de cv2
def cv2_color(frame):
    return cv2.resize(frame,(1,1))[0][0]

# Cortar imagen en base a boundingnbox, porcentaje achica el recorte
def cropp(frame, boundingbox, porcentaje = 0.2):
    x = boundingbox[0]
    y = boundingbox[1]
    w = boundingbox[2]
    d = boundingbox[3]
    
    start_x = x + int(w*porcentaje)
    end_x = x + int(w*(1-porcentaje))
    start_y = y + int(d*porcentaje)
    end_y = y + int(d*(1-porcentaje))
    return frame[start_y:end_y, start_x:end_x]

# Direccion carpeta de fotos
folder_path = 'Pruebas/Fotos/'
# Nombre imagen
image = '00055.png'

# Leer imagen
frame = cv2.imread(folder_path+image)

# Boundigbox especifico de la imagen, si se usa otra imagen se podria calcular con funciones de file_landmark
boundingbox = [224, 384, 549, 562]

# Cortar imagen
frame_cropped = cropp(frame, boundingbox)

# Mostrar en consola
color_cara_numpy = numpy_color(frame_cropped)
print('color numpy: {}'.format(color_cara_numpy))
color_cara_cv2 = np.array(cv2_color(frame_cropped))
print('color cv2: {}'.format(color_cara_cv2))

# Mostrar imagen cortada
cv2.imshow("cara", frame_cropped)
cv2.waitKey(0)
# Mostrar color numpy
cv2.imshow("color numpy", cv2.resize(np.array([[color_cara_numpy]]), (200, 200)))
cv2.waitKey(0)
# Mostrar color cv2
cv2.imshow("color cv2", cv2.resize(np.array([[color_cara_cv2]]), (200, 200)))
cv2.waitKey(0)
cv2.destroyAllWindows()