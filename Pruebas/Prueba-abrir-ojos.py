import sys
# Agregar la direccion fuera de la carpeta
sys.path.append(sys.path[0][:-8])

import procesamiento as pr
import cv2
import os
import numpy as np
import elipse

# Extraer foto a analizar
input_dir = "Pruebas/Fotos"
imagen = 0
img = cv2.imread(input_dir+"/"+os.listdir(input_dir)[imagen], 1)

#Puntos correspondiente a la foco extraidos a mano, se podria hacer desde el detected.json de la foto
puntos = [[394.60977173, 623.94610596],
            [400.59967041, 620.89489746],
            [406.58483887, 621.26568604],
            [412.18478394, 624.07305908],
            [406.59036255, 624.96112061],
            [400.53363037, 624.74304199]]

puntos = np.array(puntos)

# Calcular centroide
aux1 = np.mean(puntos, axis=0)
# Aislar puntos de los parpados
aux2 = np.concatenate((puntos[1:3],puntos[4:6]))
# Incrementar la distancia de los puntos de los parpados al centroide
aux3 = (aux2-aux1)*1.5+aux1
# Reagrupar puntos
aux4 = np.concatenate((puntos[0:1],aux3,puntos[3:4]))

# Mostrar en imagen puntos modificados
if 0:
    for x, y in puntos:
        cv2.circle(img, (int(x), int(y)), 0, (0, 255, 0), 1)

# Calcular los valores de la elipse a partir de los puntos modificados        
valores_elipse_ojoizq = elipse.get_best_ellipse_conical(aux4)
# Imprimir en consola los valores obtenidos
if 0:
    print("Centro:", valores_elipse_ojoizq['center'])
    print("Eje mayor:", valores_elipse_ojoizq['major'])
    print("Eje menor:", valores_elipse_ojoizq['major']*valores_elipse_ojoizq["ratio"])
    print("Ratio:", valores_elipse_ojoizq["ratio"])

# mostrar elipse en imagen
elipse_ojoizq = elipse.get_ellipse(valores_elipse_ojoizq['center'], valores_elipse_ojoizq['major'], valores_elipse_ojoizq["ratio"], valores_elipse_ojoizq['rotation'], 100)  
for x, y in elipse_ojoizq:
    cv2.circle(img, (int(x), int(y)), 1, (0, 255, 0), 1)

# Mostrar en imagen marcadores modificados   
for x, y in aux4:
    cv2.circle(img, (int(x), int(y)), 1, (255, 0, 0), 1)

# Usar imagen original (0) o cortada (1)
if 0:
    cv2.imshow("image", img)
    cv2.waitKey(0)
else:
    cv2.imshow("image", cv2.resize(img[610:640, 380:420], (800,600)))
    cv2.waitKey(0)

cv2.destroyAllWindows()