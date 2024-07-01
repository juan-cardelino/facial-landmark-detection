import numpy as np
import json
import elipse
import math

def norm(v):
    return np.sqrt(sum(v*v))

def dot_product(v, u):
    return sum(v*u)

def proyection(v, u):
    return dot_product(v, u)/norm(u)

# Cambio de base, renombrar como: change base
def change_base(vector, axis):
    perpendicular_axis = np.array([axis[1], -axis[0]])
    return np.array([proyection(axis, vector), proyection(perpendicular_axis, vector)])

def seno(alpha):
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
    # Get transposed array
    aux = np.array(a).T
    # Return first two columns
    return aux[0], aux[1]

def calculate_facial_feature(right_eye, left_eye, forehead, mouth):
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
    eyes_axis = np.abs(right_eye_centroid-left_eye_centroid)
    # Normalize eyes axis
    eyes_axis = eyes_axis/norm(eyes_axis)
    # Calculate eyes perpendicular axis
    p_eyes_axis = np.array([eyes_axis[1], -eyes_axis[0]])
    
    # Calculate face angle
    face_angle = math.degrees(math.atan2(eyes_axis[1], eyes_axis[0]))
    
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

