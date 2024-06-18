import cv2
import json 

# Initial setup
with open('configuracion.json') as file:
    configuracion = json.load(file)

output_dir = configuracion['path']['output_dir']
video_output = configuracion['pipeline']['camera_to_video']['video_output']
video_length = configuracion['pipeline']['camera_to_video']['video_length']
resize = (configuracion['general']['resize'][0], configuracion['general']['resize'][1])

# Get image from webcam
webcam_cap = cv2.VideoCapture(0)
# First frame
ret, frame = webcam_cap.read()
# Frame shape
h, w = frame.shape[:2]

# Initialize video format
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(output_dir+'/'+video_output, fourcc, 10, (w,h))

# Frame number
iter = 0
while webcam_cap.isOpened():
    # Get frame
    ret, frame = webcam_cap.read()
    
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
    # Show frame number
    print(iter)

# Release capture
webcam_cap.release()
out.release()
cv2.destroyAllWindows()