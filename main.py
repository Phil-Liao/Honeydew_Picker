# import OpenCV and pyplot
import cv2
from matplotlib import pyplot as plt

# read left and right images
imgR = cv2.imread('right_img.png', 0)
print(imgR.shape[:2])
imgR2 = imgR[0:3000, 0:3976]
print(imgR2.shape[:2])
imgL = cv2.imread('left_img.png', 0)
print(imgL.shape[:2])


# creates StereoBm object
stereo = cv2.StereoBM_create(numDisparities = 16,
							blockSize = 15)

# computes disparity
disparity = stereo.compute(imgL, imgR2)/16 #convert to real floatng point nums



# displays image as grayscale and plotted
plt.imshow(disparity, 'gray')
plt.show()




#depth = baseline (in mm) * focal length (in mm)/ disparity (in pixels)

