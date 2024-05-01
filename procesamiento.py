import numpy as np
import json
import cv2
import elipse
import math

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

def carga_marcadores(archivo):
    with open(archivo + ".json") as archivo:
        deteccion = json.load(archivo)
    cara = deteccion["mejor cara"]["indice"]
    cara = 0
    ojoder = np.array(deteccion["caras"][cara]["ojo derecho"])
    ojoizq = np.array(deteccion["caras"][cara]["ojo izquierdo"])
    cejader = deteccion["caras"][cara]["ceja derecha"][2:-1]
    cejaizq = deteccion["caras"][cara]["ceja izquierda"][1:-2]
    frente = np.array(cejader + cejaizq)
    labiosup = deteccion["caras"][cara]["labio superior"]
    labioinf = deteccion["caras"][cara]["labio inferior"]
    boca = np.array(labiosup+labioinf)
    boundingbox = deteccion["caras"][cara]["boundingbox"]
    return ojoder, ojoizq, frente, boca, boundingbox

def extraer_x_e_y(a):
    aux_x = aux_y = []
    for x, y in a:
        aux_x = aux_x+[x]
        aux_y = aux_y+[y]
    return np.array(aux_x), np.array(aux_y)

verbose = 1
# TODO: para que el código no te quede ilegible, podés encapsular esto en funciones
# TODO: ponelo adentro del "if verbose"
if verbose >= 1:
    ojoder, ojoizq, frente, boca, boundingbox = carga_marcadores("deteccion")
    
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
    valores_elipse_ojoder = elipse.get_best_ellipse_alt(extraer_x_e_y(ojoder))
    valores_elipse_ojoizq = elipse.get_best_ellipse_alt(extraer_x_e_y(ojoizq))
    
    #Almacenamiento estructurado
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
        "boundingbox":boundingbox
    }

    #Guardado
    with open('data.json', 'w') as file:
        json.dump(data, file, indent=4)


if verbose >= 2:
    image = cv2.imread("detected/face-detect.jpg")

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
    
# Elipse 1 https://espanol.libretexts.org/Matematicas/Algebra_lineal/%C3%81lgebra_Matricial_con_Aplicaciones_Computacionales_(Colbry)/39%3A_20_Asignaci%C3%B3n_en_clase_-_Ajuste_de_m%C3%ADnimos_cuadrados_(LSF)/39.3%3A_Ejemplo_LSF_-_Estimando_las_mejores_elipses
# Elipse 2.1 https://www.datanalytics.com/2024/02/08/ajuste-elipse/ Pagina web donde lo encontre
# Elipse 2.2 https://github.com/cjgb/ellipses/blob/dev/mylib.py Repertorio de github

if verbose >= 3:
    #Origen y ejes
    print('\n Origen y ejes')
    print('unidad', unidad)
    print('coordenadas origen', origen_ojo)
    print('proyeccion en v de origen', producto_escalar(origen_ojo, p_eje_ojos))
    
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
    print('anglo de ojo derecho', angulo_ojo_derecho*(90/np.pi))
    print('anglo de ojo izquierdo',angulo_ojo_izquierdo*(90/np.pi))