'''
This module functions cropp and rotate images
'''
import json
import cv2

def get_json_data(image, json_dir, json_suffix):
    '''
    This function takes an image name and othe setup values. It returns the image file name, the boundingbox and face angle
    
    Args:
        image (String): Image name 
        
        json_dir (String): Folder where image json file is located
        
        json_suffix (String): suffix of image json file
    
    Returns:
        image_file (String): Name of full image file name
        
        boundingbox (List): Boundignbox of the face in the image
        
        face_angle (Float): Angle of inclination of the face in the image
    '''
    # Open json file
    with open('{}/{}_{}.json'.format(json_dir, image, json_suffix)) as file:
        # Load json data
        data = json.load(file)
    # Return image file, boundingbox and face angle
    return data['image file'], data['boundingbox'], data['angulos']['cara']

def cropp(frame, boundingbox, scale_boundingbox_by = 0):
    '''
    This function takes a frame, boundigbox and a scaling factor. It sacle the bounding box by the scaling factor and cropp the image by the scaled boundingbox. It return the frame cropped
    
    Args:
        frame (Array of Arrays): Frame to cropp 
        
        boundingbox (List): Boundignbox of the face in the image
        
        scale_boundingbox_by (Float): Sacaling factor of bounding box. 
    
    Returns:
        cropped_frame (Array of Arrays): Frame cropped by the boundigbox
    '''
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
    '''
    This function takes a frame, a bounding box and a face angle. It returns the frame rotated by the face angles, using the boundingbox center as origin
    
    Args:
        frame (Array of Arrays): Frame to cropp 
        
        boundingbox (List): Boundignbox of the face in the frame, the frame is rotated by the center of the boundingbox
        
        angle (Float): Angle to rotate the frame
    
    Returns:
        rotated_frame (Array of Arrays): Frame rotated by the angle given
    
    '''
    # Get frame height and width
    height, width = frame.shape[:2]
    # Get boundingbox center
    center = (boundingbox[0]+boundingbox[2]//2,boundingbox[1]+boundingbox[3]//2)
    # Ger rotation matrix
    rotate_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    # Return frame rotated
    return cv2.warpAffine(frame, rotate_matrix, (width, height))