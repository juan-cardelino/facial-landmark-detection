# Import Packages
import cv2
import numpy as np
import procesamiento as pr
import graficar as gr

verbose = True
guardado = 1

# create an instance of the Face Detection Cascade Classifier
detector = cv2.CascadeClassifier("data/haarcascade_frontalface_alt2.xml")

# create an instance of the Facial landmark Detector with the model
landmark_detector  = cv2.face.createFacemarkLBF()
landmark_detector.loadModel("data/LFBmodel.yaml")

# get image from webcam
video_cap = cv2.VideoCapture("Output/video.avi")

ret, frame = video_cap.read()
h, w = frame.shape[:2]

if guardado < 3:
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('output/video-from-video.avi',fourcc, 60, (w,h))

iter = 0
coordenada = (int(w*.87), int(h*.95))
while video_cap.isOpened():
    # read webcam
    ret, frame = video_cap.read()
    
    if ret:

        # convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces using the haarcascade classifier on the "grayscale image"
        faces = detector.detectMultiScale(gray)
    
        try:
            # Detect landmarks on "gray"
            _, landmarks = landmark_detector.fit(gray, np.array(faces))
        except:
            landmarks = []
        print(len(landmarks))
        
        for landmark in landmarks:
            if verbose:
                centroideder, centroideizq, unidad, origen_ojo, distojos, distfrente_ojo, distboca_ojo, angulo_cara, angulo_ojo_derecho, angulo_ojo_izquierdo, valores_elipse_ojoder, valores_elipse_ojoizq = pr.calculos(landmark[0][36:42], landmark[0][42:48], landmark[0][17:27], landmark[0][48:68])
                frame = gr.ojos(frame, centroideder, centroideizq, valores_elipse_ojoder, valores_elipse_ojoizq, color = (0, 255, 0))
            frame = gr.graficar(frame, landmark[0], (255, 0, 0), int(frame.shape[1]/256))
        frame = gr.graficar_letra(frame, str(iter), coordenada, (255, 255, 255), 3)
    

        # save last instance of detected image
        if guardado > 1:
            cv2.imwrite('output/video-detect'+str(iter)+'.jpg', frame)
        if guardado < 3:
            out.write(frame) 
    
        # Show image
        cv2.imshow("frame", cv2.resize(frame,(1600,800)))

        # terminate the capture window
        if cv2.waitKey(30) & 0xFF  == ord('q'):
            break
    else: break
    iter += 1

video_cap.release()
if guardado < 3:
    out.release()
cv2.destroyAllWindows()