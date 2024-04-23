import cv2
import numpy as np

img = cv2.imread("face-detect.jpg", 0)

cv2.imwrite("face-detect-greyscale.jpg", img)

cv2.imshow("image", img)

cv2.waitKey(0)

cv2.destroyAllWindows()
