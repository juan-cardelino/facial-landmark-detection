import cv2
import numpy as np
import elipse
import procesamiento as pr

def graph_circle(frame, a, color, thickness):
    for x, y in a:
       cv2.circle(frame, (int(x), int(y)), 1, color, thickness)
    return frame

def graph_ellipse(frame, valores_elipse, color, thickness):
    x_y_elipse = elipse.get_ellipse(valores_elipse['center'], valores_elipse['major'], valores_elipse["ratio"], valores_elipse['rotation'], 100)
    frame = graph_circle(frame, x_y_elipse, color, thickness)
    return frame
   
def graph_face_section(frame, ojoder=[], ojoizq=[], boca=[], frente=[], color = (255, 0, 0)):
    m = int(frame.shape[1]/256)

    frame = graph_circle(frame, frente, color, m)
    frame = graph_circle(frame, boca, color, m)
    frame = graph_circle(frame, ojoder, color, m)
    frame = graph_circle(frame, ojoizq, color, m)
    
    return frame
    
def eyes(frame, centroideder, centroideizq, valores_elipse_ojoder, valores_elipse_ojoizq, color = (0, 255, 0)):
    m = int(frame.shape[1]/256)
    frame = graph_ellipse(frame, valores_elipse_ojoder, color, m)
    frame = graph_ellipse(frame, valores_elipse_ojoizq, color, m)
    frame = graph_circle(frame, [centroideder], color, m)
    frame = graph_circle(frame, [centroideizq], color, m)
    
    return frame

def graph_projection(frame, origin, axis, distance, color = (0, 0, 255)):
    m = int(frame.shape[1]/256)

    for i in range(distance):
        cv2.circle(frame, (int(origin[0]+axis[0]*i), int(origin[1]+axis[1]*i)), 1, color, m)

    return frame

def boundingbox(frame, boundingbox, color):
    m = int(boundingbox[2]/64)
    cv2.rectangle(frame, (boundingbox[0], boundingbox[1]), (boundingbox[0]+boundingbox[2], boundingbox[1]+boundingbox[4]), color, m)
    
    return frame

def graph_letter(frame, letra, coordenada, color, thickness):
    cv2.putText(frame, letra ,coordenada, cv2.FONT_HERSHEY_SIMPLEX , 1, color, thickness)
    return frame

def graph_axis(frame, origin, axis, length, color = (255, 255, 255)):
    aux = np.array(axis)
    aux = aux/pr.norm(aux)
    frame = graph_projection(frame, origin, aux, length, color = color)
    frame = graph_projection(frame, origin, -aux, length, color = color)
    return frame