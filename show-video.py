import cv2
import json

# Initial setup
with open('configuracion.json') as file:
    configuracion = json.load(file)

folder = configuracion['path']['output_dir']
file = configuracion['pipeline']['camera_to_video']['video_output']
resize = (configuracion['general']['resize'][0], configuracion['general']['resize'][1])

file = 'camera roll'

cap = cv2.VideoCapture('{}/{}.avi'.format(folder, file))

print("\nPress Q to release\n")

while cap.isOpened():
    # read webcam
    ret, frame = cap.read()
    
    if ret:
        # Show image
        cv2.imshow("video", cv2.resize(frame, resize))
        # terminate the capture window
        if cv2.waitKey(60) & 0xFF  == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break
    else: break