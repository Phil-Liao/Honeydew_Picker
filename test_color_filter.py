import numpy as np
import cv2
imgR = cv2.imread('right_img.png', 0)
print(imgR.shape[:2])
imgR = imgR[0:3000, 0:3976]
print(imgR.shape[:2])
imgL = cv2.imread('left_img.png', 0)
print(imgL.shape[:2])

#cap = cv2.VideoCapture(0)

while True:
    #ret, frame = cap.read()
    #width = int(cap.get(3))
    #height = int(cap.get(4))
    
    hsv = cv2.cvtColor(imgL, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    result = cv2.bitwise_and(imgL, imgR, mask=mask)
    cv2.imshow('frame', result)


    if cv2.waitKey(1) == ord('q'):
        break

#cap.release()
cv2.destroyAllWindows()


