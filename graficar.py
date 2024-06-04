import cv2
import numpy as np
import procesamiento as pr
import elipse

def graficar(frame, a, color, thickness):
    for x, y in a:
       cv2.circle(frame, (int(x), int(y)), 1, color, thickness)
    return frame

def graficar_elipse(frame, valores_elipse, color, thickness):
    x_y_elipse = elipse.get_ellipse(valores_elipse['center'], valores_elipse['major'], valores_elipse["ratio"], valores_elipse['rotation'], 100)
    frame = graficar(frame, x_y_elipse, color, thickness)
    return frame
   
def marcadores(frame, ojoder=[], ojoizq=[], boca=[], frente=[], color = (255, 0, 0)):
    m = int(frame.shape[1]/256)

    frame = graficar(frame, frente, color, m)
    frame = graficar(frame, boca, color, m)
    frame = graficar(frame, ojoder, color, m)
    frame = graficar(frame, ojoizq, color, m)
    
    return frame
    
def ojos(frame, centroideder, centroideizq, valores_elipse_ojoder, valores_elipse_ojoizq, color = (0, 255, 0)):
    m = int(frame.shape[1]/256)
    frame = graficar_elipse(frame, valores_elipse_ojoder, color, m)
    frame = graficar_elipse(frame, valores_elipse_ojoizq, color, m)
    frame = graficar(frame, [centroideder], color, m)
    frame = graficar(frame, [centroideizq], color, m)
    
    return frame

def proyecciones(frame, origen, eje, distancia, color = (0, 0, 255)):
    m = int(frame.shape[1]/256)

    for i in range(distancia):
        cv2.circle(frame, (int(origen[0]+eje[0]*i), int(origen[1]+eje[1]*i)), 1, color, m)

    return frame

def boundingbox(frame, boundingbox, color):
    m = int(boundingbox[2]/64)
    cv2.rectangle(frame, (boundingbox[0], boundingbox[1]), (boundingbox[0]+boundingbox[2], boundingbox[1]+boundingbox[4]), color, m)
    
    return frame      