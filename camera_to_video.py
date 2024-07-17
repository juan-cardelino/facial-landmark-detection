'''
This program takes a camera capture and transforms it into a video file
'''

def main():
    '''
    Run program:
    
    It starts by loading initial setup from configuracion.json
    
    Then takes camera videos and separates it by frames and show then on screen
    
    Finally saves all the frames in a video output
    '''
    import cv2
    import json 
    # Initial setup
    with open('configuracion.json') as file:
        configuracion = json.load(file)

    output_dir = configuracion['path']['output_dir']
    video_output = configuracion['pipeline']['camera_to_video']['video_output']
    video_length = configuracion['pipeline']['camera_to_video']['video_length']
    resize = configuracion['general']['resize']

    # Get image from webcam
    cap = cv2.VideoCapture(0)
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
    while cap.isOpened():
        # Get frame
        ret, frame = cap.read()
    
        if ret == True:
            # Show frame
            cv2.imshow('imagen', cv2.resize(frame, resize))
            # Save frame in video
            out.write(frame)

            # Terminate the capture window by keyboard
            if cv2.waitKey(30) & 0xFF  == ord('q'):
                break
        else: 
            break
    
        if iter > video_length:
            break
    
        iter = iter+1

    # Release capture
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return

if __name__ == "__main__":
    main()