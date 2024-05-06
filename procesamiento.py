import numpy as np
import json
import cv2
import elipse
import math
import os

def norma(a):
    return np.sqrt(sum(a*a))

def producto_escalar(a, b):
    return sum(a*b)

def cambio(vector, eje_u):
    aux = np.array([eje_u[1], -eje_u[0]])
    return np.array([sum(vector*eje_u), sum(vector*aux)])

# FIXME: poner nombres más descriptivos *y de paso, no es igual a producto_escalar?
def proyeccion(u,v):
    return producto_escalar(u,v)/norma(u)

def seno(alpha):
    return np.sin(np.arccos(alpha))

def homo_rotacion(v, cos, sen):
    v = np.array([[v[0], v[1], 1],]).T
    t = np.matrix([[cos, -sen, 0], [sen, cos, 0], [0, 0, 1]])
    w = np.array(t*v)
    w = np.array([w[0][0], w[1][0]])
    return w

def rotacion(a, cos, sen):
    aux = []
    for i in a:
        aux.append(homo_rotacion(i, cos, sen))
    return np.array(aux)

def carga_marcadores(archivo, max_caras):
    with open('Json/'+archivo + "_deteccion.json") as archivo:
        deteccion = json.load(archivo)
    cant_caras = min(deteccion["cantidad de caras"], max_caras)
    ojoder = []
    ojoizq = []
    frente = []
    boca = []
    boundingbox = []
    for i in range(cant_caras):
        ojoder.append(np.array(deteccion["caras"][i]["ojo derecho"]))
        ojoizq.append(np.array(deteccion["caras"][i]["ojo izquierdo"]))
        cejader = deteccion["caras"][i]["ceja derecha"][2:-1]
        cejaizq = deteccion["caras"][i]["ceja izquierda"][1:-2]
        frente.append(np.array(cejader + cejaizq))
        labiosup = deteccion["caras"][i]["labio superior"]
        labioinf = deteccion["caras"][i]["labio inferior"]
        boca.append(np.array(labiosup+labioinf))
        boundingbox.append(deteccion["caras"][i]["boundingbox"])
    return ojoder, ojoizq, frente, boca, boundingbox, cant_caras

def extraer_x_e_y(a):
    aux_x = []
    aux_y = []
    for x, y in a:
        aux_x = aux_x+[x]
        aux_y = aux_y+[y]
    return np.array(aux_x), np.array(aux_y)

def correcion(puntos):
    aux1 = np.mean(puntos, axis=0)
    aux2 = np.concatenate((puntos[1:3],puntos[4:6]))
    aux3 = (aux2-aux1)*1.5+aux1
    aux4 = np.concatenate((puntos[0:1],aux3,puntos[3:4]))
    return aux4

def calculos(ojoder, ojoizq, frente, boca):
    #centoride ojos
    centroideder = np.mean(ojoder, axis= 0)
    centroideizq = np.mean(ojoizq, axis= 0)
    
    distojos = norma(centroideder-centroideizq)
    unidad = (norma(ojoder[0]-ojoder[3])+norma(ojoizq[0]-ojoizq[3]))/2  # FIXME: usa nombres coherentes con la documentación
    
    #origen y ejes
    origen_ojo = (centroideder+centroideizq)/2
    eje_ojos = np.abs(centroideder-centroideizq)
    eje_ojos = eje_ojos/norma(eje_ojos)
    p_eje_ojos = np.array([eje_ojos[1], -eje_ojos[0]])
    
    #Angulo cara
    angulo_cara = math.degrees(math.atan2(eje_ojos[1],eje_ojos[0]))
    
    #Proporciones
    centrofrente = np.mean(frente, axis=0)
    centroboca = np.mean(boca, axis=0)

    distfrente_ojo = np.abs(producto_escalar(centrofrente-origen_ojo, p_eje_ojos))
    distfrente_ojo_u = np.abs(producto_escalar(centrofrente-origen_ojo, eje_ojos))
    distboca_ojo = np.abs(producto_escalar(centroboca-origen_ojo, p_eje_ojos))
    distboca_ojo_u = np.abs(producto_escalar(centroboca-origen_ojo, eje_ojos))
    
    #angulos ojos
    angulo_ojo_derecho = np.arcsin(proyeccion(ojoder[3]-ojoder[0], p_eje_ojos))
    angulo_ojo_izquierdo = np.arcsin(proyeccion(ojoizq[3]-ojoizq[0], p_eje_ojos))
    
    #Forma ojos
    valores_elipse_ojoder = elipse.get_best_ellipse_alt(ojoder)
    valores_elipse_ojoizq = elipse.get_best_ellipse_alt(ojoizq)
    return centroideder, centroideizq, unidad, origen_ojo, distojos, distfrente_ojo, distboca_ojo, angulo_cara, angulo_ojo_derecho, angulo_ojo_izquierdo, valores_elipse_ojoder, valores_elipse_ojoizq

def cuerpo(imagenes, max_caras = 1, verbose = 1, input_dir = "detected"):
    
    for imagen in imagenes:
        
        if verbose >= 1:
            nombre_j = imagen[:imagen.rfind('.')]
            ojoder, ojoizq, frente, boca, boundingbox, cant_caras = carga_marcadores(nombre_j, max_caras = max_caras)

            for i in range(cant_caras):
                centroideder, centroideizq, unidad, origen_ojo, distojos, distfrente_ojo, distboca_ojo, angulo_cara, angulo_ojo_derecho, angulo_ojo_izquierdo, valores_elipse_ojoder, valores_elipse_ojoizq = calculos(ojoder[i], ojoizq[i], frente[i], boca[i])
    
                #Almacenamiento estructurado
                if i == 0:
                    data = {
                        "puntos calculados": {
                            "ojo derecho":((centroideder-origen_ojo)/unidad).tolist(),
                            "ojo izquierdo":((centroideizq-origen_ojo)/unidad).tolist()
                        },
                        "medidas":{
                            "unidad":unidad,
                            "distancia ojos":distojos/unidad,
                            "distancia ojo-frente":distfrente_ojo/unidad,
                            "distancia ojo-boca":distboca_ojo/unidad
                        },
                        "proporcion":{
                            "frente-boca": (distboca_ojo+distfrente_ojo)/distboca_ojo
                        },
                        "angulos":{
                            "cara":angulo_cara,
                            "ojo derecho": angulo_ojo_derecho,
                            "ojo izquierdo": angulo_ojo_izquierdo
                        },
                        "forma ojos":{
                            "ratio ojo derecho":valores_elipse_ojoder["ratio"],
                            "ratio ojo izquierdo":valores_elipse_ojoizq["ratio"]
                        },
                        "boundingbox":boundingbox[i]
                    }

                    #Guardado
                    with open('Json/'+nombre_j+'_data.json', 'w') as file:
                        json.dump(data, file, indent=4)
                    print('mejor cara guardada en '+nombre_j+'.json')
        if verbose >= 2:
            image = cv2.imread(input_dir+"/"+imagen)

            #Dibujo frente y boca, solo se usa si la imagen viene vacia de copia facial 
            if 0:
                for i in frente:
                    cv2.circle(image, (int(i[0]), int(i[1])), 1, (255, 0, 0), 5)

                for i in boca:
                    cv2.circle(image, (int(i[0]), int(i[1])), 1, (255, 0, 0), 5)

            #Proyecciones de frente y boca, solo se usa para verificar
            if 0:
                for i in np.arange(int(distfrente_ojo)):
                    cv2.circle(image, (int(origen_ojo[0]+p_eje_ojos[0]*i), int(origen_ojo[1]+p_eje_ojos[1]*i)), 1, (0, 0, 255), 5)

                for i in np.arange(int(distboca_ojo)):
                    cv2.circle(image, (int(origen_ojo[0]+p_eje_ojos[0]*(-i)), int(origen_ojo[1]+p_eje_ojos[1]*(-i))), 1, (0, 0, 255), 5)

            #Ejes ojos
            if 0:
                for i in np.arange(int(distojos)):
                    cv2.circle(image, (int(centroideder[0]+eje_ojos[0]*i), int(centroideder[1]+eje_ojos[1]*i)), 1, (0, 0, 255), 5)
                aux = (ojoder[3]-ojoder[0])
                aux = aux/norma(aux)
                for i in np.arange(int(norma(ojoder[3]-ojoder[0]))):
                    cv2.circle(image, (int(ojoder[0][0]+aux[0]*i), int(ojoder[0][1]+aux[1]*i)), 1, (255, 0, 0), 5)
                aux = (ojoizq[3]-ojoizq[0])
                aux = aux/norma(aux)
                for i in np.arange(int(norma(ojoizq[3]-ojoizq[0]))):
                    cv2.circle(image, (int(ojoizq[0][0]+aux[0]*i), int(ojoizq[0][1]+aux[1]*i)), 1, (255, 0, 0), 5)
    
            #Centroides
            if 1:
                cv2.circle(image, (int(centroideder[0]), int(centroideder[1])), 1, (0, 255, 0), 5)
                cv2.circle(image, (int(centroideizq[0]), int(centroideizq[1])), 1, (0, 255, 0), 5)
                # Frente y boca
                if 0:
                    cv2.circle(image, (int(centrofrente[0]), int(centrofrente[1])), 1, (0, 255, 0), 5)
                    cv2.circle(image, (int(centroboca[0]), int(centroboca[1])), 1, (0, 255, 0), 5)
            
            if 1:
                #for i in np.concatenate((ojoder[0][1:3],ojoder[0][4:])):
                #    aux = centroideder-i
                #    aux = aux/norma(aux)
                #    for j in np.arange(int(norma(centroideder-i))):
                #        cv2.circle(image, (int(i[0]+aux[0]*j),int(i[1]+aux[1]*j)), 1, (0,0,255), 5)
                #        print(int(i[0]+aux[0]*j),int(i[1]+aux[1]*j))
                for i in ojoder[1]:
                    cv2.circle(image, (int(i[0]), int(i[1])), 1, (255, 0, 0), 5)
                
                cv2.circle(image, (int(centroideder[0]),int(centroideder[1])), 1, (0,255,0), 5)    
                #aux = (ojoder[3]-ojoder[0])
                #aux = aux/norma(aux)
                #for i in np.arange(int(norma(ojoder[3]-ojoder[0]))):
                #    cv2.circle(image, (int(ojoder[0][0]+aux[0]*i), int(ojoder[0][1]+aux[1]*i)), 1, (255, 0, 0), 5)
    
            #origen y ejes u y v
            if 0: 
                cv2.putText(image, "v" ,(int(origen_ojo[0]+p_eje_ojos[0]*25), int(origen_ojo[1]+p_eje_ojos[1]*25)), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 0, 255), 2)
                cv2.putText(image, "u" ,(int(origen_ojo[0]+eje_ojos[0]*25), int(origen_ojo[1]+eje_ojos[1]*25)), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 0, 255), 2)
                cv2.circle(image, (int(origen_ojo[0]), int(origen_ojo[1])), 1, (0, 255, 0), 5)

            #Elipse
            if 0:
                elipse_ojoder = elipse.get_ellipse(valores_elipse_ojoder['center'], valores_elipse_ojoder['major'], valores_elipse_ojoder["ratio"], valores_elipse_ojoder['rotation'], 100)
                for x, y in elipse_ojoder:
                    cv2.circle(image, (int(x), int(y)), 1, (0, 255, 0), 5)
                elipse_ojoizq = elipse.get_ellipse(valores_elipse_ojoizq['center'], valores_elipse_ojoizq['major'], valores_elipse_ojoizq["ratio"], valores_elipse_ojoizq['rotation'], 100)
                for x, y in elipse_ojoizq:
                    cv2.circle(image, (int(x), int(y)), 1, (0, 255, 0), 5)
    
            #Centroides ojos
            if 0:
                for x, y in ojoder:
                    cv2.circle(image, (int(x), int(y)), 1, (255, 0, 0), 5)

                for x, y in ojoizq:
                    cv2.circle(image, (int(x), int(y)), 1, (255, 0, 0), 5)
    
            #Boundingbox
            if 0:
                cv2.circle(image, (boundingbox[0], boundingbox[1]), 1, (0, 255, 0), 5)
                cv2.circle(image, (boundingbox[0]+boundingbox[2], boundingbox[1]+boundingbox[3]), 1, (0, 255, 0), 5)
            
            #Guardar imagen procesada
            cv2.imwrite('face-processed.jpg', image)
            #Mostrar imagen
            cv2.imshow("Image", cv2.resize(image,(900,800)))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        if verbose >= 3:
            #Origen y ejes
            print('\n Origen y ejes')
            print('unidad', unidad)
            print('coordenadas origen', origen_ojo)
            #print('proyeccion en v de origen', producto_escalar(origen_ojo, p_eje_ojos))
    
            #Centroide
            print('\n Centroides')
            print('centroide derecho', centroideder)
            print('centroide izauierdo', centroideizq)
    
            #Proporciones
            print('\n Proporciones')
            print('frente-ojo:', distfrente_ojo)
            print('boca-ojo:', distboca_ojo)
            print('frente-boca:', distfrente_ojo+distboca_ojo)
    
            #Angulos
            print('\n Angulos')
            print('anglo de ojo derecho', math.degrees(angulo_ojo_derecho))
            print('anglo de ojo izquierdo',math.degrees(angulo_ojo_izquierdo))
    return

verbose = 0
imagen = 0
cuerpo([os.listdir("detected")[imagen]], max_caras = 2, verbose = 2)