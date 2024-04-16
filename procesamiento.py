import numpy as np
import json
import cv2

def norm(a):
    return np.sqrt(a[0]*a[0]+a[1]*a[1])

def punto_recta(a, b, c):
    A = (b[1]-a[1])
    B = (a[0]-b[0])
    C = -(A*a[0]+B*a[0])
    aux = np.abs(A*c[0]+B*c[1]+C)/np.sqrt(A**2+B**2)
    return aux

def producto_escalar(a, b):
    return sum(a*b)

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
    cejader = deteccion["ceja derecha"][1:-2]
    cejaizq = deteccion["ceja izquierda"][1:-2]
    frente = np.array(cejader + cejaizq)
    labiosup = deteccion["labio superior"]
    labioinf = deteccion["labio inferior"]
    boca = np.array(labiosup+labioinf)

if 0:
    marcadores = np.genfromtxt("Marcadores.txt")
    ojoder = marcadores[36:42]
    ojoizq = marcadores[42:48]
    frente = np.array(marcadores[18:21].tolist()+marcadores[23:26].tolist())
    boca = marcadores[48:]

centroideder = np.mean(ojoder, axis= 0)
centroideizq = np.mean(ojoizq, axis= 0)

distojos = norm(centroideder-centroideizq)
print(distojos)

cos_angulo_ojos = producto_escalar(centroideder, centroideizq)/(norm(centroideder)*norm(centroideizq))
sen_angulo_ojos = seno(cos_angulo_ojos)

#ojos_rotados = rotacion(marcadores[36:48], cos_angulo_ojos, sen_angulo_ojos)

centrofrente = np.mean(frente, axis=0)

centroboca = np.mean(boca, axis=0)

distfrente_ojo = punto_recta(centroideder, centroideizq, centrofrente)

distboca_ojo = punto_recta(centroideder, centroideizq, centroboca)

# idea que encontre en linea de como hace destancia de un punto a una recta https://es.stackoverflow.com/questions/62209/distancia-entre-punto-y-segmento
#d = norm(np.cross(p2-p1, p1-p3))/norm(p2-p1)

# Eipse 1 https://espanol.libretexts.org/Matematicas/Algebra_lineal/%C3%81lgebra_Matricial_con_Aplicaciones_Computacionales_(Colbry)/39%3A_20_Asignaci%C3%B3n_en_clase_-_Ajuste_de_m%C3%ADnimos_cuadrados_(LSF)/39.3%3A_Ejemplo_LSF_-_Estimando_las_mejores_elipses

image = cv2.imread("face-detect.jpg")
cv2.circle(image, (int(centroideder[0]), int(centroideder[1])), 1, (0, 0, 255), 5)
cv2.circle(image, (int(centroideizq[0]), int(centroideizq[1])), 1, (0, 0, 255), 5)
cv2.circle(image, (int(centrofrente[0]), int(centrofrente[1])), 1, (0, 0, 255), 5)
cv2.circle(image, (int(centroboca[0]), int(centroboca[1])), 1, (0, 0, 255), 5)
cv2.imwrite('face-processed.jpg', image)
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

data = {
    "puntos calculados": {
        "ojo derecho":centroideder.tolist(),
        "ojo izquierdo":centroideizq.tolist()
    },
    "medidas":{
        "distancia ojos":distojos,
        "distancia ojo-frente":distfrente_ojo,
        "distancia ojo-boca":distboca_ojo
    },
    "proporcion":{
        "frente-boca": (distboca_ojo+distfrente_ojo)/distboca_ojo
    }
}



with open('data.json', 'w') as file:
    json.dump(data, file, indent=4)