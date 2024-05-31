import sys
sys.path.insert(1, 'C:/Users/mauri/facial-landmark-detection')

import procesamiento as pr
import cv2
import os
import numpy as np
import elipse
import json

file = 'Json'
datos = os.listdir(file)
datos2 = []
ojos = []
for i in datos:
    if i.find("deteccion") != -1:
        with open(file+'/'+i) as archivo:
            deteccion = json.load(archivo)
        if deteccion["Error"] == "No se encontraron errores":
            print(i)
            datos2.append(i)
            ojos.append(np.array(deteccion["caras"][0]["ojo derecho"]))
            ojos.append(np.array(deteccion["caras"][0]["ojo izquierdo"]))
    
#print(datos2)
    
errores = []
    
for i in ojos:
    valores_elipse_ojo_radius = pr.get_best_ellipse_radius(i, 0)
    valores_elipse_ojo_conical = elipse.get_best_ellipse_conical(i)
    if 1:
        print("")
        print("Centro")
        aux1 = np.abs(np.array(valores_elipse_ojo_conical['center'])-np.array(valores_elipse_ojo_radius['center']))/np.array(valores_elipse_ojo_conical['center'])
        print(aux1)
        print("Eje  mayor")
        print("teo", valores_elipse_ojo_conical['major'])
        print("exp", valores_elipse_ojo_radius['major'])
        aux2 = np.abs(np.array(valores_elipse_ojo_conical['major'])-np.array(valores_elipse_ojo_radius['major']))/np.array(valores_elipse_ojo_conical['major'])
        print(aux2)
        print("Eje  menor")
        aux3 = np.abs((np.array(valores_elipse_ojo_conical['major'])*np.array(valores_elipse_ojo_conical['ratio']))-(np.array(valores_elipse_ojo_radius['major'])*np.array(valores_elipse_ojo_radius['ratio'])))/(np.array(valores_elipse_ojo_conical['major'])*np.array(valores_elipse_ojo_conical["ratio"]))
        print(aux3)
        print("Ratio")
        aux4 = np.abs(np.array(valores_elipse_ojo_conical['ratio'])-np.array(valores_elipse_ojo_radius['ratio']))/np.array(valores_elipse_ojo_conical['ratio'])
        print(aux4)
        print("ratios")
        print("teo", valores_elipse_ojo_conical["ratio"])
        print("exp", valores_elipse_ojo_radius["ratio"])
        print("")
        
    errores.append([pr.norma(aux1), aux2, aux3, aux4])
if 1:
    centro = []
    eje_mayor = []
    eje_menor = []
    ratio = []
    for i in errores:
        centro.append(i[0]*100)
        eje_mayor.append(i[1]*100)
        eje_menor.append(i[2]*100)
        ratio.append(i[3]*100)
    print(errores)
    errores = np.array(errores).T*100
    centro = errores[0]
    eje_mayor = errores[1]
    eje_menor = errores[2]
    ratio = errores[3]
    print("Resumen errores en %")
    print("centro")
    print("Promedio: ",np.mean(centro))
    print("Maximo: ",max(centro))
    print("Eje mayor")
    print("Promedio: ",np.mean(eje_mayor))
    print("Maximo: ",max(eje_mayor))
    print("Eje menor")
    print("Promedio: ",np.mean(eje_menor))
    print("Maximo: ",max(eje_menor))
    print("Ratio")
    print("Promedio: ",np.mean(ratio))
    print("Maximo: ",max(ratio))
        
    print(eje_mayor)