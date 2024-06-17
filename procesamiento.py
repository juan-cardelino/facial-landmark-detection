import numpy as np
import json
import elipse
import math

def norma(a):
    return np.sqrt(sum(a*a))

def producto_escalar(a, b):
    return sum(a*b)

def cambio(vector, eje_u):
    aux = np.array([eje_u[1], -eje_u[0]])
    return np.array([sum(vector*eje_u), sum(vector*aux)])

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

def get_best_ellipse_radius(puntos, angulo):
    aux1 = np.mean(puntos, axis=0)
    aux2 = np.concatenate((puntos[0:1], puntos[3:4], [np.mean(puntos[1:3], axis=0)], [np.mean(puntos[4:6], axis=0)]))
    aux3 = aux2 - aux1
    aux4 = []
    for i in aux3:
        aux4.append(norma(i))
    
    aux5 = np.mean(aux4[0:2])
    aux6 = np.mean(aux4[2:4])
    salida = {
            'center': aux1.tolist(),
            'major': aux5,
            'ratio': aux6 / aux5,
            'rotation': angulo
        }
    return salida

def carga_marcadores(archivo, max_caras, json_dir = 'json'):
    with open('{}/{}_deteccion.json'.format(json_dir, archivo)) as archivo:
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
    return deteccion['image file'], ojoder, ojoizq, frente, boca, boundingbox, cant_caras

def extraer_x_e_y(a):
    aux = np.array(a).T
    return aux[0], aux[1]

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
    try:
        valores_elipse_ojoder = elipse.get_best_ellipse_conical(ojoder)
    except:
        valores_elipse_ojoder = get_best_ellipse_radius(ojoder, angulo_cara)
    try:
        valores_elipse_ojoizq = elipse.get_best_ellipse_conical(ojoizq)
    except:
        valores_elipse_ojoizq = get_best_ellipse_radius(ojoizq, angulo_cara)
        
    return centroideder, centroideizq, unidad, origen_ojo, distojos, distfrente_ojo, distboca_ojo, angulo_cara, angulo_ojo_derecho, angulo_ojo_izquierdo, valores_elipse_ojoder, valores_elipse_ojoizq

def calculos_alter(ojoder, ojoizq, frente, boca):
    #centoride ojos
    centroideder = np.mean(ojoder, axis= 0)
    centroideizq = np.mean(ojoizq, axis= 0)
    
    #origen y ejes
    origen_ojo = (centroideder+centroideizq)/2
    eje_ojos = np.abs(centroideder-centroideizq)
    eje_ojos = eje_ojos/norma(eje_ojos)
    p_eje_ojos = np.array([eje_ojos[1], -eje_ojos[0]])
    
    #Proporciones
    centrofrente = np.mean(frente, axis=0)
    centroboca = np.mean(boca, axis=0)
    
    return eje_ojos, p_eje_ojos, centrofrente, centroboca

def guardar_marcadores(image_file, centroideder, centroideizq, unidad, origen_ojo, distojos, distfrente_ojo, distboca_ojo, angulo_cara, angulo_ojo_derecho, angulo_ojo_izquierdo, valores_elipse_ojoder, valores_elipse_ojoizq, boundingbox, nombre_j, json_dir = "Json", json_suffix = 'data'):
    data = {
        "image file":image_file,
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
        "boundingbox":boundingbox
    }

        #Guardado
    with open(json_dir+'/'+nombre_j+'_'+json_suffix+'.json', 'w') as file:
        json.dump(data, file, indent=4)
    return

