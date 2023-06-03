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
disparity = stereo.compute(imgL, imgR2)
print(type(disparity))
# displays image as grayscale and plotted
plt.imshow(disparity, 'gray')
plt.show()

disparity_i_j = []
for i in range(len(disparity)):
    for j in disparity[i]:
        if disparity[i, j] not in disparity_i_j:
            disparity_i_j.append(disparity[i, j])
disparity_i_j.sort()
print(disparity_i_j)
for i in range(disparity_i_j[0], disparity_i_j[-1], 1):
    if i not in disparity_i_j:
        print(i, end=', ')


#depth = baseline (in mm) * focal length (in mm)/ disparity (in pixels)

