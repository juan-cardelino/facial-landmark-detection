import file_landmark as fland
import os
import procesamiento as pr

verbose = 1
imagen = 4
minimo_ancho_de_cara = 100
archivos = os.listdir("input")[3:5]

#Etapa 1 encontrar landmarks
print("")
print("Inicio etapa 1")
fland.find_landmarks(archivos, minimo_ancho_de_cara, verbose)
print("fin etapa 1")
print("")
#Etapa 2 realizar los calculos
print("Inicio etapa 2")
imagenes = os.listdir("detected")
max_caras = 1

for imagen in imagenes:
        
        if verbose >= 1:
            nombre_j = imagen[:imagen.rfind('.')]
            ojoder, ojoizq, frente, boca, boundingbox, cant_caras = pr.carga_marcadores(nombre_j, max_caras = max_caras)

            for i in range(cant_caras):
                centroideder, centroideizq, unidad, origen_ojo, distojos, distfrente_ojo, distboca_ojo, angulo_cara, angulo_ojo_derecho, angulo_ojo_izquierdo, valores_elipse_ojoder, valores_elipse_ojoizq = pr.calculos(ojoder[i], ojoizq[i], frente[i], boca[i])
                eje_ojos, p_eje_ojos, centrofrente, centroboca = pr.calculos_alter(ojoder[i], ojoizq[i], frente[i], boca[i])

                #Almacenamiento estructurado
                if i == 0:
                    pr.guardar_marcadores(centroideder, centroideizq, unidad, origen_ojo, distojos, distfrente_ojo, distboca_ojo, angulo_cara, angulo_ojo_derecho, angulo_ojo_izquierdo, valores_elipse_ojoder, valores_elipse_ojoizq, boundingbox[0], nombre_j)
                    print('mejor cara guardada en '+nombre_j+'.json')

print("fin etapa 2")
print("")

#Etapa 3 Cortar las imagenes