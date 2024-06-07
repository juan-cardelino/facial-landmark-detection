''' 
Facial Landmark Detection in Python with OpenCV

Detection from web cam
'''

# Import Packages
import cv2
import numpy as np
import procesamiento as pr
import elipse
import graficar as gr

guardado = 2

# create an instance of the Face Detection Cascade Classifier
detector = cv2.CascadeClassifier("data/haarcascade_frontalface_alt2.xml")

# create an instance of the Facial landmark Detector with the model
landmark_detector  = cv2.face.createFacemarkLBF()
landmark_detector.loadModel("data/LFBmodel.yaml")

# get image from webcam
print ("checking webcam for connection ...")
webcam_cap = cv2.VideoCapture(0)

ret, frame = webcam_cap.read()

h, w = frame.shape[:2]

if guardado < 3:
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('output/video-from-camera.avi',fourcc, 60, (w,h))

print("\nPress Q to release\n")

iter = 0
coordenada = (int(w*.87), int(h*.95))
while webcam_cap.isOpened():
    # read webcam
    ret, frame = webcam_cap.read()
    
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
        #print(len(landmarks))

        for landmark in landmarks:
            if 1:
                centroideder, centroideizq, unidad, origen_ojo, distojos, distfrente_ojo, distboca_ojo, angulo_cara, angulo_ojo_derecho, angulo_ojo_izquierdo, valores_elipse_ojoder, valores_elipse_ojoizq = pr.calculos(landmark[0][36:42], landmark[0][42:48], landmark[0][48:55], landmark[0][17:22])
                frame = gr.ojos(frame, centroideder, centroideizq, valores_elipse_ojoder, valores_elipse_ojoizq, color = (0, 255, 0)) 
            frame = gr.graficar(frame, landmark[0], (255, 0, 0), int(frame.shape[1]/256))
        frame = gr.graficar_letra(frame, str(iter), coordenada, (255, 255, 255), 3)

        if guardado > 1:
            cv2.imwrite('output/camera-detect'+str(iter)+'.jpg', frame)
        if guardado < 3:
            out.write(frame) 
        
        # Show image
        cv2.imshow("frame", cv2.resize(frame,(1600,800)))

        # terminate the capture window
        if cv2.waitKey(30) & 0xFF  == ord('q'):
            break
    else:break
    iter += 1

webcam_cap.release()
if guardado < 3:
    out.release()
cv2.destroyAllWindows()