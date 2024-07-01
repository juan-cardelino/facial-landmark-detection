# Import Packages
import cv2
import numpy as np
import procesamiento as pr
import graficar as gr
import json

# Initial setup
with open('configuracion.json') as file:
    configuracion = json.load(file)
    
calculate_feature = configuracion['pipeline']['from_video']['calculate_feature']
saving_format = configuracion['pipeline']['from_video']['saving_format']
video_file = configuracion['pipeline']['from_video']['video_file']
video_output = configuracion['pipeline']['from_video']['video_output']
video_detect = configuracion['pipeline']['from_video']['video_detect']
output_dir = configuracion['path']['output_dir']

# Create an instance of the Face Detection Cascade Classifier
detector = cv2.CascadeClassifier("data/haarcascade_frontalface_alt2.xml")

# Create an instance of the Facial landmark Detector with the model
landmark_detector  = cv2.face.createFacemarkLBF()
landmark_detector.loadModel("data/LFBmodel.yaml")

# Get image from video or camera
if video_file == 0:
    cap = cv2.VideoCapture(video_file)
else:
    cap = cv2.VideoCapture('{}.avi'.format(video_file))

# First frame
ret, frame = cap.read()
# Frame shape
h, w = frame.shape[:2]

if saving_format < 3:
    # Initialize video format
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('{}/{}.avi'.format(output_dir, video_output), fourcc, 10, (w,h))
    
print("\nPress Q to release\n")

# Frame number
iter = 0
# Frame number coordenate
coordinate = (int(w*.87), int(h*.95))
while cap.isOpened():
    # Get frame
    ret, frame = cap.read()
    
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
            if calculate_feature:
                # Calculate facial features
                centroideder, centroideizq, unidad, origen_ojo, distojos, distfrente_ojo, distboca_ojo, angulo_cara, angulo_ojo_derecho, angulo_ojo_izquierdo, valores_elipse_ojoder, valores_elipse_ojoizq = pr.calculate_facial_feature(landmark[0][36:42], landmark[0][42:48], landmark[0][17:27], landmark[0][48:68])
                # Graph facial features
                frame = gr.ojos(frame, centroideder, centroideizq, valores_elipse_ojoder, valores_elipse_ojoizq, color = (0, 255, 0))
            # Graph landmars
            frame = gr.graficar(frame, landmark[0], (255, 0, 0), int(frame.shape[1]/256))
        # Graph frame number
        frame = gr.graph_letter(frame, str(iter), coordinate, (255, 255, 255), 3)

        # Save frame
        if saving_format > 1:
            # Save frame on image
            cv2.imwrite('{}/{}_{}.jpg'.format(output_dir, video_detect, str(iter)), frame)
        if saving_format < 3:
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
cap.release()
if saving_format < 3:
    out.release()
cv2.destroyAllWindows()