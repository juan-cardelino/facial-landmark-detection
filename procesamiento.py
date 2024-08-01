
'''
This module main functions use the landmarks to calculte all facial features and save them in a json file. The other functions are algebraic operations use to calculate facial feature
'''
import numpy as np
import json
import elipse
import math

def norm(v):
    '''
    This algorith takes a vector as v and returns its two-dimensional norm
    
    Args:
        v (Array): Vector
        
    Returns:
        norm (int): Norm of v
    '''
    return np.sqrt(sum(v*v))

def dot_product(v, u):
    '''
    This function takes two vectors as u and v and returns the dot product of the two vectors
    
    Args:
        v (Array): First vector
        
        u (Array): Second vector
        
    Returns:
        dot_product (int): Dot product of v and u
    '''
    return sum(v*u)

def proyection(v, u):
    '''
    This function takes two vectors as v and u and returns the proyection of v in u
    
    Args:
        v (Array): Vector to proyect
        
        u (Array): Vector where to proyect
        
    Returns:
        proyection (int): Proyection of v in u
    '''
    return dot_product(v, u)/norm(u)

# Cambio de base, renombrar como: change base
def change_base(vector, axis):
    '''
    This function takes a vector and a axis, and return the proyections of the vector in the axis and its perpendicular counterpart
    
    Args:
        vector (Array): Vector to proyect in new base
        
        axis (Array): Axis of new base
        
    Returns:
        new_vector (Array): Vector in new base
    '''
    perpendicular_axis = np.array([axis[1], -axis[0]])
    return np.array([proyection(axis, vector), proyection(perpendicular_axis, vector)])

def seno(alpha):
    '''
    Args:
        alpha (Float): Cosine of an angle
    
    Returns:
        sin (Float): Sin of the angle
    '''
    return np.sin(np.arccos(alpha))

# Este no se si fletarlo, no se usa
def homo_rotation(v, cos, sen):
    v = np.array([[v[0], v[1], 1],]).T
    t = np.matrix([[cos, -sen, 0], [sen, cos, 0], [0, 0, 1]])
    w = np.array(t*v)
    w = np.array([w[0][0], w[1][0]])
    return w

# Este tambien se podria fletar
def rotation(a, cos, sen):
    aux = []
    for i in a:
        aux.append(homo_rotation(i, cos, sen))
    return np.array(aux)

def get_best_ellipse_radius(points, angle):
    '''
    This function takes a 6 points array and an angle. It calculate the parameters of the ellipse that best fits base on the radius from the point centroid. This parameters are return in a dict
    
    Args:
        points (list): List of points 
        
        angle (float): Angle of rotation of ellipse
    
    Returns:
        output (dict): Dict with center, mayor axis, ratio and rotation
    '''
    # Get points centroid
    centroid = np.mean(points, axis=0)
    # Reorganize point
    reorganized_points = np.concatenate((points[0:1], points[3:4], [np.mean(points[1:3], axis=0)], [np.mean(points[4:6], axis=0)]))
    # Get point-centroid distance
    distance = reorganized_points - centroid
    # Create norm list
    norms = []
    for i in distance:
        # Load norm list
        norms.append(norm(i))
    
    # Calculate mayor axis
    mayor_axis = np.mean(norms[0:2])
    # Calculate minor axis
    minor_axis = np.mean(norms[2:4])
    # Create output dict
    output = {
            'center': centroid.tolist(),
            'major': mayor_axis,
            'ratio': minor_axis / mayor_axis,
            'rotation': angle
        }
    return output

def load_landmarks(json_file, max_faces, json_dir = 'json'):
    '''
    This function takes the name and direction of json file as json_file and json_dir, also takes the amount of faces to unpack as max_faces (unpack minimun between max_faces and faces able in json). It returns the image file name, arrays with landmarks of right_eye, left_eye, forehead, mouth and boundingbox of every face, also returns the amount of faces unpacked
    
    Args:
        json_file (String): Name of json file
        
        max_faces (int): Number of max faces to process
        
        json_dir (String): Name of the folder where to load json file
        
    Returns:
        file_name (String): Name of the image file
        
        right_eye (list): List of right eyes points
        
        left_eye (list): List of left eyes points
        
        forehead (list): List of forehead points
        
        mouth (list): List of mouth points
        
        boundingbox (list): List of boundingbox parameters
        
        face_amount (int): Minimun between max_faces and faces detected
    '''
    with open('{}/{}_deteccion.json'.format(json_dir, json_file)) as file:
        detection = json.load(file)
    # Calculate faces to get landmarks
    face_amount = min(detection["cantidad de caras"], max_faces)
    # Create output lists
    right_eye = []
    left_eye = []
    forehead = []
    mouth = []
    boundingbox = []
    for i in range(face_amount):
        # Load output lists
        right_eye.append(np.array(detection["caras"][i]["ojo derecho"]))
        left_eye.append(np.array(detection["caras"][i]["ojo izquierdo"]))
        right_eyebrow = detection["caras"][i]["ceja derecha"][2:-1]
        left_eyebrow = detection["caras"][i]["ceja izquierda"][1:-2]
        forehead.append(np.array(right_eyebrow + left_eyebrow))
        sup_lip = detection["caras"][i]["labio superior"]
        inf_lip = detection["caras"][i]["labio inferior"]
        mouth.append(np.array(sup_lip+inf_lip))
        boundingbox.append(detection["caras"][i]["boundingbox"])
    return detection['image file'], right_eye, left_eye, forehead, mouth, boundingbox, face_amount

def get_x_y(a):
    '''
    This function takes as input a two dimensional array and return the first two columns
    
    Args:
        a (2D Array): Array of points
    
    Returns:
        colum1 (Array): First colum of array
        
        colum2 (Array): Second colum of array
    '''
    # Get transposed array
    aux = np.array(a).T
    # Return first two columns
    return aux[0], aux[1]

def calculate_facial_feature(right_eye, left_eye, forehead, mouth):
    '''
    This function takes the right eye, left eye, forehead and mouth landmarks to calculate all facial features
    
    Args:
        right_eye (list): List of points of right eye
        
        left_eye (list): List of points of left eye
        
        forehead (list): List of points of forehead
        
        mouth (list): List of points of mouth
        
    Returns:
        right_eye_centroid (tuple): Right eye centroid point
        
        left_eye_centroid (tuple): Left eye centroid point
        
        unit (Float): Face unit
        
        eyes_origin (tuple): Eyes origin point
        
        eye_distance (Float): Eyes centroids distance
        
        forhead_eyes_distance (Float): Eyes-Forehead distance
        
        mouth_eyes_distance (Float): Eyes-Mouth distance
        
        face_angle (Float): Face angle
        
        right_eye_angle (Float): Right eye angle
        
        left_eye_angle (Float): Left eye angle
        
        ellipse_values_right_eye (dict): Dict of values of right eye ellipse
        
        ellipse_values_left_eye (dict): Dict of values of left eye ellipse
    '''
    # Calculate eye centroids
    right_eye_centroid = np.mean(right_eye, axis= 0)
    left_eye_centroid = np.mean(left_eye, axis= 0)
    
    # Calculate eyes centorids distance
    eye_distance = norm(right_eye_centroid-left_eye_centroid)
    # Calculate face unit from eyes
    unit = (norm(right_eye[0]-right_eye[3])+norm(left_eye[0]-left_eye[3]))/2
    
    # Calculate eyes origin from centroids
    eyes_origin = (right_eye_centroid+left_eye_centroid)/2
    # Calculate eyes axis
    eyes_axis = right_eye_centroid-left_eye_centroid
    # Calculate face angle
    face_angle = math.degrees(math.atan2(eyes_axis[1], -eyes_axis[0]))
    # Normalize eyes axis
    eyes_axis = np.abs(eyes_axis)
    eyes_axis = eyes_axis/norm(eyes_axis)
    # Calculate eyes perpendicular axis
    p_eyes_axis = np.array([eyes_axis[1], -eyes_axis[0]])
    
    
    
    # Calculate forehead and mouth centroids
    forehead_centroid = np.mean(forehead, axis=0)
    mouth_centroid = np.mean(mouth, axis=0)

    # Calculate forehead eyes distance
    forhead_eyes_distance = np.abs(proyection(forehead_centroid-eyes_origin, p_eyes_axis))
    forhead_eyes_distance_u = np.abs(proyection(forehead_centroid-eyes_origin, eyes_axis))
    # Calculate mouth eyes distance
    mouth_eyes_distance = np.abs(proyection(mouth_centroid-eyes_origin, p_eyes_axis))
    mouth_eyes_distance_u = np.abs(proyection(mouth_centroid-eyes_origin, eyes_axis))
    
    # Eyes angle
    right_eye_angle = np.arcsin(proyection(p_eyes_axis, right_eye[3]-right_eye[0]))
    left_eye_angle = np.arcsin(proyection(p_eyes_axis, left_eye[3]-left_eye[0]))
    
    # Eyes shape
    try:
        ellipse_values_right_eye = elipse.get_best_ellipse_conical(get_x_y(right_eye))
    except:
        ellipse_values_right_eye = get_best_ellipse_radius(right_eye, face_angle)
    try:
        ellipse_values_left_eye = elipse.get_best_ellipse_conical(get_x_y(left_eye))
    except:
        ellipse_values_left_eye = get_best_ellipse_radius(left_eye, face_angle)
        
    return right_eye_centroid, left_eye_centroid, unit, eyes_origin, eye_distance, forhead_eyes_distance, mouth_eyes_distance, face_angle, right_eye_angle, left_eye_angle, ellipse_values_right_eye, ellipse_values_left_eye

def calculate_auxiliar(right_eye, left_eye, forehead, mouth):
    '''
    Args:
        right_eye (list): List of points of right eye
        
        left_eye (list): List of points of left eye
        
        forehead (list): List of points of forehead
        
        mouth (list): List of points of mouth
    
    Returns:
        eyes_axis (tuple): Eyes axis
        
        p_eyes_axis (tuple): Eyes perpendicular axis
        
        forehead_centroid (tuple): Forehead centroid point
        
        mouth_centroid (tuple): Mouth centroid point
    '''
    
    # Calculate eyes centroids
    right_eye_centroid = np.mean(right_eye, axis= 0)
    left_eye_centroid = np.mean(left_eye, axis= 0)
    
    # Calculate eyes origin from centroids
    eyes_origin = (right_eye_centroid+left_eye_centroid)/2
    # Calculate eyes axis
    eyes_axis = np.abs(right_eye_centroid-left_eye_centroid)
    # Normalize eyes axis
    eyes_axis = eyes_axis/norm(eyes_axis)
    # Calculate eyes perpendicular axis
    p_eyes_axis = np.array([eyes_axis[1], -eyes_axis[0]])
    
    # Calculate forehead and mouth centroids
    forehead_centroid = np.mean(forehead, axis=0)
    mouth_centroid = np.mean(mouth, axis=0)
    
    return eyes_axis, p_eyes_axis, forehead_centroid, mouth_centroid

def save_features(image_file, right_eye_centroid, left_eye_centroid, unit, eyes_origin, eye_distance, forhead_eyes_distance, mouth_eyes_distance, face_angle, right_eye_angle, left_eye_angle, ellipse_values_right_eye, ellipse_values_left_eye, boundingbox, json_name, json_dir = "Json", json_suffix = 'data'):
    '''
    This function takes the calculated facial features and save them in a json file
    
    Args:
        image_file (String): Image file name
        
        right_eye_centroid (tuple): Right eye centroid point
        
        left_eye_centroid (tuple): Left eye centroid point
        
        unit (Float): Face unit
        
        eyes_origin (tuple): Eyes origin point
        
        eye_distance (Float): Eyes centroids distance
        
        forhead_eyes_distance (Float): Eyes-Forehead distance
        
        mouth_eyes_distance (Float): Eyes-Mouth distance
        
        face_angle (Float): Face angle
        
        right_eye_angle (Float): Right eye angle
        
        left_eye_angle (Float): Left eye angle
        
        ellipse_values_right_eye (dict): Dict of values of right eye ellipse
        
        ellipse_values_left_eye (dict): Dict of values of left eye ellipse
        
        boundingbox (list): List of boundingbox parameters
        
        json_name (String): Name of the json file
        
        json_dir (Sting): Name of json folder
        
        json_suffix (Sting): Suffix of json name
    '''
    # Create data dict
    data = {
        "image file":image_file,
        "puntos calculados": {
            "ojo derecho":((right_eye_centroid-eyes_origin)/unit).tolist(),
            "ojo izquierdo":((left_eye_centroid-eyes_origin)/unit).tolist()
        },
        "medidas":{
            "unidad":unit,
            "distancia ojos":eye_distance/unit,
            "distancia ojo-frente":forhead_eyes_distance/unit,
            "distancia ojo-boca":mouth_eyes_distance/unit
        },
        "proporcion":{
            "frente-boca": (mouth_eyes_distance+forhead_eyes_distance)/mouth_eyes_distance
        },
        "angulos":{
            "cara":face_angle,
            "ojo derecho": right_eye_angle,
            "ojo izquierdo": left_eye_angle
        },
        "forma ojos":{
            "ratio ojo derecho":ellipse_values_right_eye["ratio"],
            "ratio ojo izquierdo":ellipse_values_left_eye["ratio"]
        },
        "boundingbox":boundingbox
    }

    # Save data dict in json
    with open('{}/{}_{}.json'.format(json_dir, json_name, json_suffix), 'w') as file:
        json.dump(data, file, indent=4)
    return

