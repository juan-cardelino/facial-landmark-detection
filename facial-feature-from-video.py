# Import Packages
import cv2
import numpy as np
import procesamiento as pr
import elipse


# create an instance of the Face Detection Cascade Classifier
detector = cv2.CascadeClassifier("data/haarcascade_frontalface_alt2.xml")

# create an instance of the Facial landmark Detector with the model
landmark_detector  = cv2.face.createFacemarkLBF()
landmark_detector.loadModel("data/LFBmodel.yaml")

# get image from webcam
print ("checking webcam for connection ...")
webcam_cap = cv2.VideoCapture("video.avi")


while(True):
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
        print(len(landmarks))
        if 1:
            for i in landmarks:
                centroideder=np.mean(i[0][36:42], axis=0)
                centroideder, centroideizq, unidad, origen_ojo, distojos, distfrente_ojo, distboca_ojo, angulo_cara, angulo_ojo_derecho, angulo_ojo_izquierdo, valores_elipse_ojoder, valores_elipse_ojoizq = pr.calculos(i[0][36:42], i[0][42:48], i[0][48:55], i[0][17:22])
                cv2.circle(frame, (int(centroideder[0]), int(centroideder[1])), 1, (0, 255, 0), 2)
                cv2.circle(frame, (int(centroideizq[0]), int(centroideizq[1])), 1, (0, 255, 0), 2)
                cv2.circle(frame, (int(origen_ojo[0]), int(origen_ojo[1])), 1, (0, 255, 0), 2)
                elipse_ojoder = elipse.get_ellipse(valores_elipse_ojoder['center'], valores_elipse_ojoder['major'], valores_elipse_ojoder["ratio"], valores_elipse_ojoder['rotation'], 100)
                for x, y in elipse_ojoder:
                    cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), 5)
                elipse_ojoizq = elipse.get_ellipse(valores_elipse_ojoizq['center'], valores_elipse_ojoizq['major'], valores_elipse_ojoizq["ratio"], valores_elipse_ojoizq['rotation'], 100)
                for x, y in elipse_ojoizq:
                    cv2.circle(frame, (int(x), int(y)), 1, (0, 255, 0), 5)
                #cv2.circle(frame, (int(centroideder[0]), int(centroideder[1])), 1, (0, 255, 0), 2)

        for landmark in landmarks:
            for x,y in landmark[0]:
                # display landmarks on "frame/image,"
                # with blue colour in BGR and thickness 2
                cv2.circle(frame, (int(x), int(y)), 1, (255, 0, 0), int(frame.shape[1]/256))
    

        # save last instance of detected image
        cv2.imwrite('video-detect.jpg', frame)    
    
        # Show image
        cv2.imshow("frame", cv2.resize(frame,(1600,800)))

        # terminate the capture window
        if cv2.waitKey(20) & 0xFF  == ord('q'):
            webcam_cap.release()
            cv2.destroyAllWindows()
            break
    else: break