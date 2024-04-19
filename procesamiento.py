import numpy as np
import json
import cv2

def norma(a):
    return np.sqrt(a[0]*a[0]+a[1]*a[1])

def punto_recta(a, b, c):
    A = (b[1]-a[1])
    B = (a[0]-b[0])
    C = -(A*a[0]+B*a[0])
    aux = np.abs(A*c[0]+B*c[1]+C)/np.sqrt(A**2+B**2)
    return aux

def producto_escalar(a, b):
    return sum(a*b)

#def perpendicular(v, u):
#    v1 = v/norma(v)
#    u1 = u - (sum(v1*u)*v1)
#    return v1, u1/norma(u1)

#def cambio(vector, eje_u):
#    aux = np.array([[eje_u[0], eje_u[1]],[eje_u[1], -eje_u[0]]])
#    aux2 = np.array([vector,]).T
#    print(aux)
#    print(vector)
#    print(aux2)
#   return aux*aux2.T

def cambio(vector, eje_u):
    aux = np.array([eje_u[1], -eje_u[0]])
    return np.array([sum(vector*eje_u), sum(vector*aux)])

def pr(u,v):
    return sum(u*v)

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

if 1:
    with open("deteccion.json") as archivo:
        deteccion = json.load(archivo)
    
    ojoder = np.array(deteccion["ojo derecho"])
    ojoizq = np.array(deteccion["ojo izquierdo"])
    cejader = deteccion["ceja derecha"][2:-1]
    cejaizq = deteccion["ceja izquierda"][1:-2]
    frente = np.array(cejader + cejaizq)
    labiosup = deteccion["labio superior"]
    labioinf = deteccion["labio inferior"]
    boca = np.array(labiosup+labioinf)
else:
    marcadores = np.genfromtxt("Marcadores.txt")
    ojoder = marcadores[36:42]
    ojoizq = marcadores[42:48]
    frente = np.array(marcadores[18:21].tolist()+marcadores[23:26].tolist())
    boca = marcadores[48:]

centroideder = np.mean(ojoder, axis= 0)
centroideizq = np.mean(ojoizq, axis= 0)
origen_ojo = (centroideder+centroideizq)/2
unidad = (norma(ojoder[0]-ojoder[3])+norma(ojoizq[0]-ojoizq[3]))/2

print(unidad)

distojos = norma(centroideder-centroideizq)

centrofrente = np.mean(frente, axis=0)
centroboca = np.mean(boca, axis=0)

eje_ojos = np.abs(centroideder-centroideizq)
eje_ojos = eje_ojos/norma(eje_ojos)
p_eje_ojos = np.array([eje_ojos[1], -eje_ojos[0]])

distfrente_ojo = np.abs(pr(centrofrente-origen_ojo, p_eje_ojos))
distfrente_ojo_u = np.abs(pr(centrofrente-origen_ojo, eje_ojos))
distboca_ojo = np.abs(pr(centroboca-origen_ojo, p_eje_ojos))
distboca_ojo_u = np.abs(pr(centroboca-origen_ojo, eje_ojos))

#print('origen')
#print(origen_ojo)
#print(pr(origen_ojo, p_eje_ojos))
#print(centroideder)
#print(centroideizq)

print('frente:', distfrente_ojo)
print('boca:', distboca_ojo)
print('frente+boca:', distfrente_ojo+distboca_ojo)

#angulos ojos

angulo_ojo_derecho = np.arcsin(pr(ojoder[3]-ojoder[0], p_eje_ojos)/norma(ojoder[3]-ojoder[0]))
angulo_ojo_izquierdo = np.arcsin(pr(ojoizq[3]-ojoizq[0], p_eje_ojos)/norma(ojoizq[3]-ojoizq[0]))

print(angulo_ojo_derecho*(90/np.pi))
print(angulo_ojo_izquierdo*(90/np.pi))

# Eipse 1 https://espanol.libretexts.org/Matematicas/Algebra_lineal/%C3%81lgebra_Matricial_con_Aplicaciones_Computacionales_(Colbry)/39%3A_20_Asignaci%C3%B3n_en_clase_-_Ajuste_de_m%C3%ADnimos_cuadrados_(LSF)/39.3%3A_Ejemplo_LSF_-_Estimando_las_mejores_elipses

image = cv2.imread("face-detect.jpg")

if 0:
    for i in frente:
     cv2.circle(image, (int(i[0]), int(i[1])), 1, (255, 0, 0), 5)

    for i in boca:
        cv2.circle(image, (int(i[0]), int(i[1])), 1, (255, 0, 0), 5)

if 0:
    for i in np.arange(int(distfrente_ojo)):
        #cv2.circle(image, (int(origen_ojo[0]+eje_ojos[0]*i), int(origen_ojo[1]+eje_ojos[1]*i)), 1, (255, 0, 255), 5)
        cv2.circle(image, (int(origen_ojo[0]+p_eje_ojos[0]*i), int(origen_ojo[1]+p_eje_ojos[1]*i)), 1, (0, 0, 255), 5)

    for i in np.arange(int(distboca_ojo)):
        #cv2.circle(image, (int(origen_ojo[0]+eje_ojos[0]*i), int(origen_ojo[1]+eje_ojos[1]*i)), 1, (0, 0, 255), 5)
        cv2.circle(image, (int(origen_ojo[0]+p_eje_ojos[0]*(-i)), int(origen_ojo[1]+p_eje_ojos[1]*(-i))), 1, (0, 0, 255), 5)

if 1:
    for i in np.arange(int(distojos)):
        #cv2.circle(image, (int(origen_ojo[0]+eje_ojos[0]*i), int(origen_ojo[1]+eje_ojos[1]*i)), 1, (255, 0, 255), 5)
        cv2.circle(image, (int(centroideder[0]+eje_ojos[0]*i), int(centroideder[1]+eje_ojos[1]*i)), 1, (0, 0, 255), 5)
    algo = (ojoder[3]-ojoder[0])
    algo = algo/norma(algo)
    for i in np.arange(int(norma(ojoder[3]-ojoder[0]))):
        #cv2.circle(image, (int(origen_ojo[0]+eje_ojos[0]*i), int(origen_ojo[1]+eje_ojos[1]*i)), 1, (255, 0, 255), 5)
        cv2.circle(image, (int(ojoder[0][0]+algo[0]*i), int(ojoder[0][1]+algo[1]*i)), 1, (0, 125, 255), 5)
    algo = (ojoizq[3]-ojoizq[0])
    algo = algo/norma(algo)
    for i in np.arange(int(norma(ojoizq[3]-ojoizq[0]))):
        #cv2.circle(image, (int(origen_ojo[0]+eje_ojos[0]*i), int(origen_ojo[1]+eje_ojos[1]*i)), 1, (255, 0, 255), 5)
        cv2.circle(image, (int(ojoizq[0][0]+algo[0]*i), int(ojoizq[0][1]+algo[1]*i)), 1, (0, 125, 255), 5)
    
cv2.circle(image, (int(centroideder[0]), int(centroideder[1])), 1, (0, 255, 0), 5)
cv2.circle(image, (int(centroideizq[0]), int(centroideizq[1])), 1, (0, 255, 0), 5)
#cv2.circle(image, (int(centrofrente[0]), int(centrofrente[1])), 1, (0, 255, 0), 5)
#cv2.circle(image, (int(centroboca[0]), int(centroboca[1])), 1, (0, 255, 0), 5)
#cv2.putText(image, "v" ,(int(origen_ojo[0]+p_eje_ojos[0]*25), int(origen_ojo[1]+p_eje_ojos[1]*25)), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 0, 255), 2)
#cv2.putText(image, "u" ,(int(origen_ojo[0]+eje_ojos[0]*25), int(origen_ojo[1]+eje_ojos[1]*25)), cv2.FONT_HERSHEY_SIMPLEX , 1, (0, 0, 255), 2)
cv2.circle(image, (int(origen_ojo[0]), int(origen_ojo[1])), 1, (0, 255, 0), 5)

cv2.imwrite('face-processed.jpg', image)
cv2.imshow("Image", cv2.resize(image,(900,800)))
cv2.waitKey(0)
cv2.destroyAllWindows()

data = {
    "puntos calculados": {
        "ojo derecho":((centroideder-origen_ojo)/unidad).tolist(),
        "ojo izquierdo":((centroideizq-origen_ojo)/unidad).tolist()
    },
    "medidas":{
        "distancia ojos":distojos/unidad,
        "distancia ojo-frente":distfrente_ojo/unidad,
        "distancia ojo-boca":distboca_ojo/unidad
    },
    "proporcion":{
        "frente-boca": (distboca_ojo+distfrente_ojo)/distboca_ojo
    },
    "angulos":{
        "ojo derecho": angulo_ojo_derecho,
        "ojo izquierdo": angulo_ojo_izquierdo
    }
}



with open('data.json', 'w') as file:
    json.dump(data, file, indent=4)


