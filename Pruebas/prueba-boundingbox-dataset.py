import json
import numpy as np

# Abrir dataset
with open('ffhq-dataset-v2.json') as archivo:
        ffhq_data = json.load(archivo)

# Extraer landmarks
landmarks = np.array(ffhq_data['0']["image"]["face_landmarks"])

# Mostrar landmarks
print('Landmarks \n{}'.format(landmarks))

# Mostrar primer landmark
print('\nPrimer landmark: {}'.format(landmarks[0]))

# Mostrar primer columna, segunda analoga
print('Primer columna: {}'.format(landmarks.T[0]))

# Maximo y minimo de primer columna, segunda analoga
print('Maximo de la columna: {}'.format(max(landmarks.T[0])))
print('Minimo de la columna: {}'.format(min(landmarks.T[0])))

# Mostrar boundingbox, formato [(min(x), min(y)), (max(x), max(y))]
print('\nBoundingbox: {}\n'.format([(min(landmarks.T[0]), min(landmarks.T[0])), (min(landmarks.T[1]), min(landmarks.T[1]))]))