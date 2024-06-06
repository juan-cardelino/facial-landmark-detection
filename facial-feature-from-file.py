import file_landmark as fland
import os
import procesamiento as pr

verbose = 1
minimo_ancho_de_cara = 100
raw_input = "input"
archivos = os.listdir(raw_input)
detected_output = "detected"

#Etapa 1 encontrar landmarks
print("")
print("Inicio etapa 1")
fland.find_landmarks(archivos, minimo_ancho_de_cara, verbose, raw_input, detected_output)
print("fin etapa 1")
print("")
#Etapa 2 realizar los calculos
print("Inicio etapa 2")
imagenes = []
for i in os.listdir("Json")[:2]:
    # Se identifica si el json es de deteccion, si es se continua
    if i[-14:] == "deteccion.json":
        imagenes.append(i[:-15])
max_caras = 1

for imagen in imagenes:
        
        ojoder, ojoizq, frente, boca, boundingbox, cant_caras = pr.carga_marcadores(imagen, max_caras = max_caras)

        for i in range(cant_caras):
            centroideder, centroideizq, unidad, origen_ojo, distojos, distfrente_ojo, distboca_ojo, angulo_cara, angulo_ojo_derecho, angulo_ojo_izquierdo, valores_elipse_ojoder, valores_elipse_ojoizq = pr.calculos(ojoder[i], ojoizq[i], frente[i], boca[i])
            #eje_ojos, p_eje_ojos, centrofrente, centroboca = pr.calculos_alter(ojoder[i], ojoizq[i], frente[i], boca[i])

            #Almacenamiento estructurado
            if i == 0:
                pr.guardar_marcadores(centroideder, centroideizq, unidad, origen_ojo, distojos, distfrente_ojo, distboca_ojo, angulo_cara, angulo_ojo_derecho, angulo_ojo_izquierdo, valores_elipse_ojoder, valores_elipse_ojoizq, boundingbox[0], imagen)
                print('mejor cara guardada en '+imagen+'.json')

print("fin etapa 2")
print("")

#Etapa 3 Cortar las imagenes