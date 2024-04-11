import numpy as np



marcadores = np.genfromtxt("Marcadores.txt")
#print(marcadores[0][0])

ojoder = marcadores[36:42]
ojoizq = marcadores[42:48]

#print(ojoderecho)
#print(ojoizquierdo)

centroideder = np.mean(ojoder, axis= 0)
centroideizq = np.mean(ojoizq, axis= 0)

print(centroideder)
print(centroideizq)