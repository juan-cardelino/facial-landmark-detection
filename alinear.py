import json
import cv2

def extraer_datos_json(imagen, json_dir, json_suffix):
    with open(json_dir+'/'+imagen+'_'+json_suffix+'.json') as file:
        data = json.load(file)
    return data['image file'], data['boundingbox'], data['angulos']['cara']

def cropp(frame, boundingbox, porcentaje = 0):
        x = boundingbox[0]
        y = boundingbox[1]
        w = boundingbox[2]
        d = boundingbox[3]
    
        start_x = x + int(w*porcentaje)
        end_x = x + int(w*(1-porcentaje))
        start_y = y + int(d*porcentaje)
        end_y = y + int(d*(1-porcentaje))
        return frame[start_y:end_y, start_x:end_x]

def rotate(frame, boundingbox, angulo):
        height, width = frame.shape[:2] 
        center = (boundingbox[0]+boundingbox[2]//2,boundingbox[1]+boundingbox[3]//2)
        rotate_matrix = cv2.getRotationMatrix2D(center, angulo, 1)
        return cv2.warpAffine(frame, rotate_matrix, (width, height))