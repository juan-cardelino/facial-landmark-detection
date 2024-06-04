''' 
Facial Landmark Detection in Python with OpenCV

Detection from web cam
'''

# Import Packages
import cv2
import numpy as np


# create an instance of the Face Detection Cascade Classifier
detector = cv2.CascadeClassifier("data/haarcascade_frontalface_alt2.xml")

# create an instance of the Facial landmark Detector with the model
landmark_detector  = cv2.face.createFacemarkLBF()
landmark_detector.loadModel("data/LFBmodel.yaml")

# get image from webcam
print ("checking webcam for connection ...")
webcam_cap = cv2.VideoCapture(0)


while(True):
    # read webcam
    _, frame = webcam_cap.read()

    # convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using the haarcascade classifier on the "grayscale image"
    faces = detector.detectMultiScale(gray)
    

    for (x,y,w,d) in faces:
        # Detect landmarks on "gray"
        _, landmarks = landmark_detector.fit(gray, np.array(faces))
        
        lista = landmarks[0]
        print(len(landmarks))

        for landmark in landmarks:
            for x,y in landmark[0]:
                # display landmarks on "frame/image,"
                # with blue colour in BGR and thickness 2
                cv2.circle(frame, (int(x), int(y)), 1, (255, 0, 0), 2)

    # save last instance of detected image
    cv2.imwrite('Output/face-detect.jpg', frame)    
    
    # Show image
    cv2.imshow("frame", cv2.resize(frame,(1600,800)))

    # terminate the capture window
    if cv2.waitKey(20) & 0xFF  == ord('q'):
        webcam_cap.release()
        cv2.destroyAllWindows()
        break
