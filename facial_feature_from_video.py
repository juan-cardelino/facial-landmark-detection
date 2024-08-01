'''
This program calculate facial features from a video capture and save them in another video
'''

# Import Packages
import cv2
import numpy as np
import procesamiento as pr
import graficar as gr
import json

def main():
    '''
    It starts by loading initial setup from configuracion.json

    The program separates the video capture by frames
    
    For each frame it detects if there is any face, if there is, it calculates the facial features. 
    
    All frames are save in an output video and/or independent images, depending on the configuracion.json saving_format.
    '''
    # Initial setup
    with open('configuracion.json') as file:
        configuration = json.load(file)

    output_dir = configuration['path']['output_dir']
    models_dir = configuration['path']['model_dir'] 
    calculate_feature = configuration['pipeline']['from_video']['calculate_feature']
    saving_format = configuration['pipeline']['from_video']['saving_format']
    video_file = configuration['pipeline']['from_video']['video_file']
    video_output = configuration['pipeline']['from_video']['video_output']
    video_detect = configuration['pipeline']['from_video']['video_detect']
    resize = configuration['general']['resize']
    haarcascade = configuration['general']['face detection model']
    LBFmodel = configuration['general']['landmark detection model']

    # Create an instance of the Face Detection Cascade Classifier
    detector = cv2.CascadeClassifier("{}/{}".format(models_dir, haarcascade))

    # Create an instance of the Facial landmark Detector with the model
    landmark_detector  = cv2.face.createFacemarkLBF()
    landmark_detector.loadModel("{}/{}".format(models_dir, LBFmodel))

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
                    right_centroid, left_centroid, unit, eye_origin, eye_distance, eye_forhead_distance, ete_mouth_distance, face_angle, right_eye_angle, left_eye_angle, right_eye_ellipse_values, left_eye_ellipse_values = pr.calculate_facial_feature(landmark[0][36:42], landmark[0][42:48], landmark[0][17:27], landmark[0][48:68])
                    # Graph facial features
                    frame = gr.graph_eyes(frame, right_centroid, left_centroid, right_eye_ellipse_values, left_eye_ellipse_values, color = (0, 255, 0))
                # Graph landmars
                frame = gr.graph_circle(frame, landmark[0], (255, 0, 0), int(frame.shape[1]/256))
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
            cv2.imshow("frame", cv2.resize(frame, resize))

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

if __name__ == "__main__":
    main()