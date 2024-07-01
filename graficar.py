import cv2
import numpy as np
import elipse
import procesamiento as pr

def graph_circle(frame, a, color, thickness):
    for x, y in a:
       cv2.circle(frame, (int(x), int(y)), 1, color, thickness)
    return frame

def graph_ellipse(frame, ellipse_values, color, thickness):
    x_y_elipse = elipse.get_ellipse(ellipse_values['center'], ellipse_values['major'], ellipse_values["ratio"], ellipse_values['rotation'], 100)
    frame = graph_circle(frame, x_y_elipse, color, thickness)
    return frame
   
def graph_face_section(frame, right_eye=[], left_eye=[], mouth=[], forehead=[], color = (255, 0, 0)):
    m = int(frame.shape[1]/256)

    frame = graph_circle(frame, forehead, color, m)
    frame = graph_circle(frame, mouth, color, m)
    frame = graph_circle(frame, right_eye, color, m)
    frame = graph_circle(frame, left_eye, color, m)
    
    return frame
    
def graph_eyes(frame, right_eye_centroid, left_eye_centroid, rigth_eye_ellipse_values, left_eye_ellipse_values, color = (0, 255, 0)):
    m = int(frame.shape[1]/256)
    frame = graph_ellipse(frame, rigth_eye_ellipse_values, color, m)
    frame = graph_ellipse(frame, left_eye_ellipse_values, color, m)
    frame = graph_circle(frame, [right_eye_centroid], color, m)
    frame = graph_circle(frame, [left_eye_centroid], color, m)
    
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

def graph_letter(frame, letter, coordenate, color, thickness):
    cv2.putText(frame, letter ,coordenate, cv2.FONT_HERSHEY_SIMPLEX , 1, color, thickness)
    return frame

def graph_axis(frame, origin, axis, length, color = (255, 255, 255)):
    aux = np.array(axis)
    aux = aux/pr.norm(aux)
    frame = graph_projection(frame, origin, aux, length, color = color)
    frame = graph_projection(frame, origin, -aux, length, color = color)
    return frame