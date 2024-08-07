'''
This program takes a video and estimates the roll angle of the faces present in the video
'''
# Import Packages
import cv2
import numpy as np
import graficar as gr
import json
import math

def main():
    '''
    Run code:
    
    It starts by loading initial setup from configuracion.json

    Then program calculates facial landmarks
    
    Finally calculates de roll angle of the face
    '''
    # Initial setup
    with open('configuracion.json') as file:
        configuracion = json.load(file)
        
    calculate_feature = configuracion['pipeline']['from_video']['calculate_feature']
    output_dir = configuracion['path']['output_dir']
    model_dir = configuracion['path']['model_dir']
    cap_input = configuracion['pipeline']['camera_control']['cap_input']
    video_output = configuracion['pipeline']['camera_control']['roll_output']
    face_detection_model = configuracion['general']['face detection model']
    landmark_detection_model = configuracion['general']['landmark detection model']
    resize = configuracion['general']['resize']

    # Create an instance of the face detection Cascade Classifier
    detector = cv2.CascadeClassifier('{}/{}'.format(model_dir, face_detection_model))

    # Create an instance of the facial landmark detector with the model
    landmark_detector  = cv2.face.createFacemarkLBF()
    landmark_detector.loadModel('{}/{}'.format(model_dir, landmark_detection_model))

    # Get image from video
    cap = cv2.VideoCapture(cap_input)

    # First frame
    ret, frame = cap.read()
    # Frame shape
    h, w = frame.shape[:2]

    # Initialize video format
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('{}/{}.avi'.format(output_dir, video_output), fourcc, 10, (w,h))

    # Show break key
    print("\nPress Q to release\n")

    # Frame number
    iter = 0
    # Frame number coordenate
    coordinate = (int(w*.87), int(h*.95))
    angles = []
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
                # Face centroid
                centroid = np.mean(landmark[0], axis=0)
                # Graph landmarks
                frame = gr.graph_circle(frame, landmark[0], (255, 0, 0), int(frame.shape[1]/256))
                # Length axis
                l_axis = int(frame.shape[1]/4)
                # Graph axis 90
                frame = gr.graph_axis(frame, centroid, [0, 1], l_axis)
                # Graph axis 30
                frame = gr.graph_axis(frame, centroid, [1, 2], l_axis)
                # Graph axis 30
                frame = gr.graph_axis(frame, centroid, [1, -2], l_axis)
                # Graph face centroid
                frame = gr.graph_circle(frame, [centroid], (0, 0, 0), int(frame.shape[1]/64))
                
                # Calculate eyes centroids
                right_eye_centroid = np.mean(landmark[0][36:42].tolist(), axis=0)
                left_eye_centroid = np.mean(landmark[0][42:48].tolist(), axis=0)
                
                # Calculate eyes axis
                eyes_axis = right_eye_centroid-left_eye_centroid
                # Calculate face angle
                face_angle = math.degrees(math.atan2(eyes_axis[1], -eyes_axis[0]))
                
                angles.append(face_angle)
                
            # Graph frame number
            frame = gr.graph_letter(frame, str(iter), coordinate, (255, 255, 255), 3)
            
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
    out.release()
    cv2.destroyAllWindows()
    
    print('\nMax roll: {}\nMin roll: {}'.format(max(angles), min(angles)))
    return

if __name__ == "__main__":
    main()