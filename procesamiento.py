import numpy as np



marcadores = np.genfromtxt("Marcadores.txt")
#print(marcadores[0][0])

ojoder = marcadores[36:42]
ojoizq = marcadores[42:48]

#print(ojoderecho)
#print(ojoizquierdo)

centroideder = np.mean(ojoder, axis= 0)
centroideizq = np.mean(ojoizq, axis= 0)

frente = marcadores[17:27]
centrofrente = np.mean(frente, axis=0)

boca = marcadores[48:]
centroboca = np.mean(boca, axis=0)

# idea que encontre en linea de como hace destancia de un punto a una recta https://es.stackoverflow.com/questions/62209/distancia-entre-punto-y-segmento
#d = norm(np.cross(p2-p1, p1-p3))/norm(p2-p1)