import cv2
import numpy as np
import os
import elipse

def extraer_x_e_y(a):
    aux_x = aux_y = []
    for x, y in a:
        aux_x = aux_x+[x]
        aux_y = aux_y+[y]
    return np.array(aux_x), np.array(aux_y)

def norma(a):
    return np.sqrt(sum(a*a))

def coreccion(points):
    puntos = []
    for i in points[0]:
        puntos.append([points[0][i], points[1][i]])
    aux1 = np.mean(puntos, axis=0)
    aux2 = puntos[1:3]+puntos[4:6]
    normas = []
    for j in aux2:
        normas.append(norma(aux1-j))
    norma_m = np.mean(normas)
    aux3 = []
    for k in aux2:
        aux3.append(list(aux1+(aux1-k)*(1.5*norma_m/norma(aux1-k))))
    aux4 = puntos[0:1]+aux3+puntos[3:4]
    aux_x = aux_y = []
    for x, y in aux4:
        aux_x = aux_x+[x]
        aux_y = aux_y+[y]
    return elipse.fit_ellipse(np.array(aux_x), np.array(aux_y))

imagen = 2
os.listdir("input")[imagen]

img = cv2.imread("input/"+os.listdir("input")[imagen], 1)

puntos = [[936.9950561523438,637.3349609375],
            [942.6434326171875,632.6384887695312],
            [948.8971557617188,632.423828125],
            [954.689453125,634.623046875],
            [949.6332397460938,636.6954956054688],
            [943.4136352539062,637.2012329101562]]

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
    for x, y in aux4:
        cv2.circle(img, (int(x), int(y)), 0, (0, 255, 0), 1)

#valores_elipse_ojoizq = elipse.get_best_ellipse_alt(puntos)
valores_elipse_ojoizq = elipse.get_best_ellipse_alt(aux4)

#Aparentemente el orden de los puntos genera diferencia en el codigo, porque? ni idea

if 1:
    elipse_ojoizq = elipse.get_ellipse(valores_elipse_ojoizq['center'], valores_elipse_ojoizq['major'], valores_elipse_ojoizq["ratio"], valores_elipse_ojoizq['rotation'], 100)
    for x, y in elipse_ojoizq:
        cv2.circle(img, (int(x), int(y)), 1, (0, 255, 0), 1)
if 1:    
    for x, y in aux4:
        cv2.circle(img, (int(x), int(y)), 1, (255, 0, 0), 1)



if 0:
    cv2.imshow("image", img)
    cv2.waitKey(0)


print(img.shape[:2])

image_cropped = img[610:640, 380:420]
#cv2.imwrite("face-detect-greyscale.jpg", img)

if 1:
    cv2.imshow("image", cv2.resize(image_cropped, (800,600)))

    cv2.waitKey(0)

cv2.destroyAllWindows()

algo = [1, 2, 3]
for i in algo:
    print(algo)