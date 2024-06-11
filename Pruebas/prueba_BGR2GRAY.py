import cv2
import numpy as np

verbose = 1

# Abrir una imagen a color cualquiera
frame = cv2.imread("input/input2.png")

if verbose:
    # Mostrar imagen a color
    cv2.imshow("imagen color", frame)
    cv2.waitKey(0)

# Pasar a grayscale (escala de grises)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

if verbose:
    # Mostrar escala grayscale
    cv2.imshow("imagen gris", gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Mostrar informacion del primer pixel de cada imagen
print('Codificacion BGR 3 puntos: {}'.format(frame[0,0]))
print('Promedio de puntos: {}'.format(np.mean(frame[0,0])))
print('Codificacion grayscale un punto: {}'.format(gray[0,0]))

# Grayscale es distinto a promediar los colores de BGR