import cv2
import json

# Initial setup
with open('configuracion.json') as file:
    configuracion = json.load(file)

folder = configuracion['path']['output_dir']
file = configuracion['pipeline']['camera_to_video']['video_output']
resize = (configuracion['general']['resize'][0], configuracion['general']['resize'][1])

webcam_cap = cv2.VideoCapture(folder+"/"+file)

print("\nPress Q to release\n")

while webcam_cap.isOpened():
    # read webcam
    ret, frame = webcam_cap.read()
    
    if ret:
        # Show image
        cv2.imshow("video", cv2.resize(frame, resize))
        # terminate the capture window
        if cv2.waitKey(60) & 0xFF  == ord('q'):
            webcam_cap.release()
            cv2.destroyAllWindows()
            break
    else: break