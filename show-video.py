import cv2
import json

# Initial setup
with open('configuracion.json') as file:
    configuration = json.load(file)

folder = configuration['path']['output_dir']
file = configuration['pipeline']['camera_to_video']['video_output']
resize = configuration["general"]["resize"]

# Get capture
cap = cv2.VideoCapture('{}/{}.avi'.format(folder, file))

print("\nPress Q to release\n")

while cap.isOpened():
    # Read capture
    ret, frame = cap.read()
    
    if ret:
        # Show image
        cv2.imshow("video", cv2.resize(frame, resize))
        # Terminate the capture window
        if cv2.waitKey(60) & 0xFF  == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break
    else: break