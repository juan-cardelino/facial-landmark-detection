import sys
sys.path.insert(1, 'C:/Users/mauri/facial-landmark-detection')

import procesamiento as pr
import cv2
import os
import numpy as np
import elipse

input_dir = "Pruebas/Fotos"
imagen = 0
img = cv2.imread(input_dir+"/"+os.listdir(input_dir)[imagen], 1)

puntos = [[394.60977173, 623.94610596],
            [400.59967041, 620.89489746],
            [406.58483887, 621.26568604],
            [412.18478394, 624.07305908],
            [406.59036255, 624.96112061],
            [400.53363037, 624.74304199]]

puntos = np.array(puntos)

aux1 = np.mean(puntos, axis=0)
aux2 = np.concatenate((puntos[1:3],puntos[4:6]))
aux3 = (aux2-aux1)*1.5+aux1
aux4 = np.concatenate((puntos[0:1],aux3,puntos[3:4]))

if 0:
    for x, y in puntos:
        cv2.circle(img, (int(x), int(y)), 0, (0, 255, 0), 1)
        
valores_elipse_ojoizq = elipse.get_best_ellipse_conical(aux4)
if 0:
    print(valores_elipse_ojoizq['center'])
    print(valores_elipse_ojoizq['major'])
    print(valores_elipse_ojoizq['major']*valores_elipse_ojoizq["ratio"])
    print(valores_elipse_ojoizq["ratio"])

elipse_ojoizq = elipse.get_ellipse(valores_elipse_ojoizq['center'], valores_elipse_ojoizq['major'], valores_elipse_ojoizq["ratio"], valores_elipse_ojoizq['rotation'], 100)  
for x, y in elipse_ojoizq:
    cv2.circle(img, (int(x), int(y)), 1, (0, 255, 0), 1)

# Mostrar marcadores modificados   
for x, y in aux4:
    cv2.circle(img, (int(x), int(y)), 1, (255, 0, 0), 1)

# Usar imagen original o cortada
if 0:
    cv2.imshow("image", img)
    cv2.waitKey(0)
else:
    cv2.imshow("image", cv2.resize(img[610:640, 380:420], (800,600)))
    cv2.waitKey(0)

cv2.destroyAllWindows()