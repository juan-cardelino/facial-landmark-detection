import cv2

webcam_cap = cv2.VideoCapture(0)

ret, frame = webcam_cap.read()

h, w = frame.shape[:2]

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter('video.avi',fourcc, 60, (w,h))

iter = 0
while webcam_cap.isOpened():
    ret, frame = webcam_cap.read()
    
    if ret == True:
        cv2.imshow('imagen', frame)
        out.write(frame)
    
        if cv2.waitKey(20) & 0xFF  == ord('q'):
            break
    else: 
        break
    
    if iter > 300:
        break
    
    iter = iter+1
    print(iter)

webcam_cap.release()
out.release()
cv2.destroyAllWindows()