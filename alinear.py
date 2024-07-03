import json
import cv2

def get_json_data(image, json_dir, json_suffix):
    # Open json file
    with open('{}/{}_{}.json'.format(json_dir, image, json_suffix)) as file:
        # Load json data
        data = json.load(file)
    # Return image file, boundingbox and face angle
    return data['image file'], data['boundingbox'], data['angulos']['cara']

def cropp(frame, boundingbox, scale_boundingbox_by = 0):
        # Get boundingbox corners coordenates
        x = boundingbox[0]
        y = boundingbox[1]
        w = boundingbox[2]
        d = boundingbox[3]

        # Get coordenates to cropp
        start_x = x + int(w*scale_boundingbox_by)
        end_x = x + int(w*(1-scale_boundingbox_by))
        start_y = y + int(d*scale_boundingbox_by)
        end_y = y + int(d*(1-scale_boundingbox_by))
        # Return frame cropped
        return frame[start_y:end_y, start_x:end_x]

def rotate(frame, boundingbox, angle):
        # Get frame height and width
        height, width = frame.shape[:2]
        # Get boundingbox center
        center = (boundingbox[0]+boundingbox[2]//2,boundingbox[1]+boundingbox[3]//2)
        # Ger rotation matrix
        rotate_matrix = cv2.getRotationMatrix2D(center, angle, 1)
        # Return frame rotated
        return cv2.warpAffine(frame, rotate_matrix, (width, height))