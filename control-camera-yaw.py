# Import Packages
import cv2
import numpy as np
import procesamiento as pr
import graficar as gr
import json

def graph_eje(frame, origin, axis, length, color = (255, 255, 255)):
    aux = np.array(axis)
    aux = aux/pr.norma(aux)
    frame = gr.proyecciones(frame, origin, aux, length, color = color)
    frame = gr.proyecciones(frame, origin, -aux, length, color = color)
    return frame
    

# Initial setup
with open('configuracion.json') as file:
    configuracion = json.load(file)
    
calculate_feature = configuracion['pipeline']['from_video']['calculate_feature']
saving_format = configuracion['pipeline']['from_video']['saving_format']
video_file = 0
video_output = configuracion['pipeline']['camera control']['pitch output']
video_detect = configuracion['pipeline']['from_video']['video_detect']
output_dir = configuracion['path']['output_dir']

# Create an instance of the Face Detection Cascade Classifier
detector = cv2.CascadeClassifier("data/haarcascade_frontalface_alt2.xml")

# Create an instance of the Facial landmark Detector with the model
landmark_detector  = cv2.face.createFacemarkLBF()
landmark_detector.loadModel("data/LFBmodel.yaml")

# Get image from video
video_cap = cv2.VideoCapture(video_file)

# First frame
ret, frame = video_cap.read()
# Frame shape
h, w = frame.shape[:2]

# Initialize video format
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('{}/{}.avi'.format(output_dir, video_output), fourcc, 10, (w,h))

print("\nPress Q to release\n")

# Frame number
iter = 0
# Frame number coordenate
coordinate = (int(w*.87), int(h*.95))
# Create list of angles
angles = []
while video_cap.isOpened():
    # Get frame
    ret, frame = video_cap.read()
    
    if ret:

        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces using the haarcascade classifier on the "grayscale image"
        faces = detector.detectMultiScale(gray)
    
        try:
            # Detect landmarks on "gray"
            _, landmarks = landmark_detector.fit(gray, np.array(faces))
        except:
            landmarks = []
        #print(len(landmarks))
        
        for landmark in landmarks:
            # Face centroid
            centroid = np.mean(landmark[0], axis=0)
            # Graph landmars
            frame = gr.graficar(frame, landmark[0], (255, 0, 0), int(frame.shape[1]/256))
            # Graph face centroid
            frame = gr.graficar(frame, [centroid], (0, 0, 0), int(frame.shape[1]/64))
            
            # Get x and y 
            x, y = pr.extraer_x_e_y(landmark[0])
            
            # Max x
            max_x = max(x)
            # Min x
            min_x = min(x)
            # Graph max x, min x and centroid
            graph_eje(frame, [max_x, 0], [0, 1], int(frame.shape[1]), color = (255, 255, 255))
            graph_eje(frame, [min_x, 0], [0, 1], int(frame.shape[1]), color = (255, 255, 255))
            graph_eje(frame, [centroid[0], 0], [0, 1], int(frame.shape[1]), color = (255, 255, 255))
            
            # Calculate angle
            angles.append(max_x+min_x-2*centroid[0])
        
        # Graph frame number
        frame = gr.graficar_letra(frame, str(iter), coordinate, (255, 255, 255), 3)
        
        # Save frame on video
        out.write(frame) 
    
        # Show frame
        cv2.imshow("frame", cv2.resize(frame,(1600,800)))

        # Terminate the capture window
        if cv2.waitKey(30) & 0xFF  == ord('q'):
            break
    else: break
    iter += 1

# Release capture
video_cap.release()
out.release()
cv2.destroyAllWindows()

print('\nMax yaw: {}\nMin yaw: {}'.format(max(angles), min(angles)))