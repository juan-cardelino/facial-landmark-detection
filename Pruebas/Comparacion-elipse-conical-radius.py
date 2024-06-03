import sys
# Agregar la direccion fuera de la carpeta
sys.path.append(sys.path[0][:-8])

import procesamiento as pr
import os
import numpy as np
import elipse
import json

# Funcion de calculo de error relativo
def error_relativo(a, b):
    auxa = np.array(a)
    auxb = np.array(b)
    return abs(auxa-auxb)*100/auxa

# Extraer jsons
file = 'Json'
datos = os.listdir(file)
# Lista donde guardar los ojos
ojos = []
# se recorren los jsons
for i in datos:
    # Se identifica si el json es de deteccion, si es se continua
    if i.find("deteccion") != -1:
        # Se abre el archivo y se extraen los datos
        with open(file+'/'+i) as archivo:
            deteccion = json.load(archivo)
        # Si la deteccion no tiene error se guardan los ojos
        if deteccion["Error"] == "No se encontraron errores":
            print(i)
            # Se guardan los ojos sin distinguir lado
            ojos.append(np.array(deteccion["caras"][0]["ojo derecho"]))
            ojos.append(np.array(deteccion["caras"][0]["ojo izquierdo"]))

# Lista de errores relativos   
errores = []
    
for i in ojos:
    # Calculo de valores de elipse exprimental on angulo = 0 (se considera generico)
    valores_elipse_ojo_radius = pr.get_best_ellipse_radius(i, 0)
    # Calculo de valores de elipse teorico
    try: # Existen casos en la que el codigo de ellipse_conical se rompe
        valores_elipse_ojo_conical = elipse.get_best_ellipse_conical(i)
    except: # En esos casos se utiliza dos veces el codigo ellipse_radius
        valores_elipse_ojo_radius = pr.get_best_ellipse_radius(i, 0)
    
    # Se calcula el error relativo de los valores (entro, eje mayor, eje menor y ratio)
    aux1 = error_relativo(valores_elipse_ojo_conical['center'], valores_elipse_ojo_radius['center'])
    aux2 = error_relativo(valores_elipse_ojo_conical['major'], valores_elipse_ojo_radius['major'])
    aux3 = error_relativo(np.array(valores_elipse_ojo_conical['major'])*np.array(valores_elipse_ojo_conical['ratio']), np.array(valores_elipse_ojo_radius['major'])*np.array(valores_elipse_ojo_radius['ratio']))
    aux4 = error_relativo(valores_elipse_ojo_conical['ratio'], valores_elipse_ojo_radius['ratio'])
    if 1:
        # Se desglosan todos los datos de los ojos y sus errores
        print("")
        print("Centro")
        print("Error: {}%".format(round(np.mean(aux1), 2)))
        print("Error: {}%".format(aux1))
        print("teo", valores_elipse_ojo_conical['center'])
        print("exp", valores_elipse_ojo_radius['center'])
        print("Eje  mayor")
        print("Error: {}%".format(round(aux2, 2)))
        print("teo", valores_elipse_ojo_conical['major'])
        print("exp", valores_elipse_ojo_radius['major'])
        print("Eje  menor")
        print("Error: {}%".format(round(aux3, 2)))
        print("teo", np.array(valores_elipse_ojo_conical['major'])*np.array(valores_elipse_ojo_conical['ratio']))
        print("exp", np.array(valores_elipse_ojo_radius['major'])*np.array(valores_elipse_ojo_radius['ratio']))
        print("Ratio")
        print("Error: {}%".format(round(aux4, 2)))
        print("teo", valores_elipse_ojo_conical["ratio"])
        print("exp", valores_elipse_ojo_radius["ratio"])
        print("")
    
    # Se guardan los errores relevantes   
    errores.append([pr.norma(aux1), aux2, aux3, aux4])
# Se extraen los errores
errores = np.array(errores).T
centro = errores[0]
eje_mayor = errores[1]
eje_menor = errores[2]
ratio = errores[3]
# se imprime el resumen de los errores
print("")
print("RESUMEN ERRORES\n")
print("centro")
print("Promedio: {}%".format(round(np.mean(centro),2)))
print("Maximo: {}%".format(round(max(centro), 2)))
print("Eje mayor")
print("Promedio: {}%".format(round(np.mean(eje_mayor),2)))
print("Maximo: {}%".format(round(max(eje_mayor), 2)))
print("Eje menor")
print("Promedio: {}%".format(round(np.mean(eje_menor),2)))
print("Maximo: {}%".format(round(max(eje_menor), 2)))
print("Ratio")
print("Promedio: {}%".format(round(np.mean(ratio),2)))
print("Maximo: {}%".format(round(max(ratio), 2)))