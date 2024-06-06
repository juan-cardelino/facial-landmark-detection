import cv2

folder = "output"
file = "video-from-camera.avi"

webcam_cap = cv2.VideoCapture(folder+"/"+file)

print("\nPress Q to release\n")

while webcam_cap.isOpened():
    # read webcam
    ret, frame = webcam_cap.read()
    
    if ret:
        # Show image
        cv2.imshow("video", cv2.resize(frame,(1600,800)))
        # terminate the capture window
        if cv2.waitKey(60) & 0xFF  == ord('q'):
            webcam_cap.release()
            cv2.destroyAllWindows()
            break
    else: break