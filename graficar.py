'''
This module takes frames and graph basic objects in it
'''

import cv2
import numpy as np
import elipse
import procesamiento as pr

def graph_circle(frame, a, color, thickness):
    '''
    This function takes a frame, a list of points as a, a color and a tickness. It graphs all the point with the given color and thickness in the frame. It return the frame
    
    Args:
        frame (Array of Array): Frame to change
        
        a (list): List of points to graph in frame
        
        color (tuple): BGR color of points to graph
        
        thickness (int): tickness of point to graph
    
    Returns:
        frame_changed (Array of Array): Frame with points graphed
    '''
    for x, y in a:
       cv2.circle(frame, (int(x), int(y)), 1, color, thickness)
    return frame

def graph_ellipse(frame, ellipse_values, color, thickness):
    '''
    This function takes a frame, ellipse values, a color and thicknees. It return the frame with the ellipse graphed
    
    Args:
        frame (Array of Array): Frame to change
        
        ellipse_values (dict): Dict of ellipse values
        
        color (tuple): BGR color of points to graph
        
        thickness (int): tickness of point to graph
    
    Returns:
        frame_changed (Array of Array): Frame with ellipse graphed
    '''
    x_y_elipse = elipse.get_ellipse(ellipse_values['center'], ellipse_values['major'], ellipse_values["ratio"], ellipse_values['rotation'], 100)
    frame = graph_circle(frame, x_y_elipse, color, thickness)
    return frame
   
def graph_face_section(frame, right_eye=[], left_eye=[], mouth=[], forehead=[], color = (255, 0, 0)):
    '''
    This function takes a frame, face sections (up to four) as array and a color. It return the frame with all sections graphed
    
    Args:
        frame (Array of Array): Frame to change
        
        right_eye (list): List of right eye points to graph in frame
        
        left_eye (list): List of left eye points to graph in frame
        
        mouth (list): List of mouth points to graph in frame
        
        forehead (list): List of forehead points to graph in frame
        
        color (tuple): BGR color of points to graph
    
    Returns:
        frame_changed (Array of Array): Frame with sections graphed
    '''
    m = int(frame.shape[1]/256)

    frame = graph_circle(frame, forehead, color, m)
    frame = graph_circle(frame, mouth, color, m)
    frame = graph_circle(frame, right_eye, color, m)
    frame = graph_circle(frame, left_eye, color, m)
    
    return frame
    
def graph_eyes(frame, right_eye_centroid, left_eye_centroid, right_eye_ellipse_values, left_eye_ellipse_values, color = (0, 255, 0)):
    '''
    This function takes a frame, eyes centroids, eyes ellipse values and a color. It return the frame with the centoroids and ellipse graphed
    
    Args:
        frame (Array of Array): Frame to change
        
        right_eye (tuple): Right eye centroid point to graph in frame
        
        left_eye (tuple): Left eye centroid point to graph in frame
        
        right_eye_ellipse_values (dict): Dict of values of right eye ellipse to graph in frame
        
        left_eye_ellipse_values (dict): Dict of values of left eye ellipse to graph in frame
        
        color (tuple): BGR color of points to graph
    
    Returns:
        frame_changed (Array of Array): Frame with eyes graphed
    '''
    m = int(frame.shape[1]/256)
    frame = graph_ellipse(frame, right_eye_ellipse_values, color, m)
    frame = graph_ellipse(frame, left_eye_ellipse_values, color, m)
    frame = graph_circle(frame, [right_eye_centroid], color, m)
    frame = graph_circle(frame, [left_eye_centroid], color, m)
    
    return frame

def graph_projection(frame, origin, axis, distance, color = (0, 0, 255)):
    '''
    This function takes a frame, an origin point, axis, distance and color. It returns the frame with a line from origin to distance in axis direction graphed
    
    Args:
        frame (Array of Array): Frame to change
        
        origin (tuple): Origin point for axis
        
        axis (tuple): Direction of axis
        
        distance (int): Length of axis
        
        color (tuple): BGR color of points to graph
    
    Returns:
        frame_changed (Array of Array): Frame with axis graphed
    '''
    m = int(frame.shape[1]/256)

    for i in range(distance):
        cv2.circle(frame, (int(origin[0]+axis[0]*i), int(origin[1]+axis[1]*i)), 1, color, m)

    return frame

def boundingbox(frame, boundingbox, color):
    '''
    This function takes a frame boundingbox and color. It returns a frame with the boundingbox graphed
    
    Args:
        frame (Array of Array): Frame to change
        
        boundingbox (list): List of boundingbox parameters
        
        color (tuple): BGR color of points to graph
    
    Returns:
        frame_changed (Array of Array): Frame with boundingbox graphed
    '''
    m = int(boundingbox[2]/64)
    cv2.rectangle(frame, (boundingbox[0], boundingbox[1]), (boundingbox[0]+boundingbox[2], boundingbox[1]+boundingbox[3]), color, m)
    
    return frame

def graph_letter(frame, letter, coordenate, color, thickness):
    '''
    This function takes a frame, letter, coordenate, color and thickness. It returns the frame with the letter graphed in the coordenate
    
    Args:
        frame (Array of Array): Frame to change
        
        letter (Stirng): String to graph in frame
        
        coordenate (tuple): Coordenate of where to graph letter
        
        color (tuple): BGR color of points to graph
        
        thickness (int): tickness of point to graph
    
    Returns:
        frame_changed (Array of Array): Frame with letter graphed
    '''
    cv2.putText(frame, letter ,coordenate, cv2.FONT_HERSHEY_SIMPLEX , 1, color, thickness)
    return frame

def graph_axis(frame, origin, axis, length, color = (255, 255, 255)):
    '''
    This function takes a frame, an origin point, axis, length and color. It returns the frame with a line from lenght in -axis direction to lenght in axis direction graphed
    
    Args:
        frame (Array of Array): Frame to change
        
        origin (tuple): Origin point for axis
        
        axis (tuple): Direction of axis
        
        distance (int): Length of axis
        
        color (tuple): BGR color of points to graph
    
    Returns:
        frame_changed (Array of Array): Frame with axis graphed
    '''
    aux = np.array(axis)
    aux = aux/pr.norm(aux)
    frame = graph_projection(frame, origin, aux, length, color = color)
    frame = graph_projection(frame, origin, -aux, length, color = color)
    return frame