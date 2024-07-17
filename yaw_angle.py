'''
This program takes a video and estimates the yaw angle of the faces present in the video
'''
# Import Packages
import cv2
import numpy as np
import procesamiento as pr
import graficar as gr
import json

def main():
    '''
    It starts by loading initial setup from configuracion.json.

    Then program calculates facial landmarks
    
    Then it calculates the face centroid and the face boundingbox center
    
    Finally distance between this two points is use to calculate the yaw angle
    '''    
    # Initial setup
    with open('configuracion.json') as file:
        configuration = json.load(file)

    output_dir = configuration['path']['output_dir']
    model_dir = configuration['path']['model_dir']   
    calculate_feature = configuration['pipeline']['from_video']['calculate_feature']
    cap_input = configuration['pipeline']['camera control']['cap_input']
    video_output = configuration['pipeline']['camera control']['yaw_output']
    face_detection_model = configuration['general']['face detection model']
    landmark_detection_model = configuration['general']['landmark detection model']
    resize = configuration['general']["resize"]

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

    print("\nPress Q to release\n")

    # Frame number
    iter = 0
    # Frame number coordenate
    coordinate = (int(w*.87), int(h*.95))
    # Create list of angles
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
                # Graph landmars
                frame = gr.graph_circle(frame, landmark[0], (255, 0, 0), int(frame.shape[1]/256))
                # Graph face centroid
                frame = gr.graph_circle(frame, [centroid], (0, 0, 0), int(frame.shape[1]/64))
                
                # Get x and y 
                x, y = pr.get_x_y(landmark[0])
                
                # Max x
                max_x = max(x)
                # Min x
                min_x = min(x)
                # Graph max x, min x and centroid
                frame = gr.graph_axis(frame, [max_x, 0], [0, 1], int(frame.shape[1]), color = (255, 255, 255))
                frame = gr.graph_axis(frame, [min_x, 0], [0, 1], int(frame.shape[1]), color = (255, 255, 255))
                frame = gr.graph_axis(frame, [centroid[0], 0], [0, 1], int(frame.shape[1]), color = (255, 255, 255))
                
                # Calculate angle from arctangent
                opuesto = max_x+min_x-2*centroid[0] # Calclate opposite side
                adyacente = (max_x-min_x)/2 # Calculate adjacent
                angle = round(np.arctan(opuesto/adyacente)*180/np.pi, 1)
                # Show angle
                print(angle)
                # Save angle in angles list
                angles.append(angle)
            
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

    print('\nMax yaw: {}\nMin yaw: {}'.format(max(angles), min(angles)))
    return

if __name__ == "__main__":
    main()