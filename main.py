# import OpenCV and pyplot
import cv2
from matplotlib import pyplot as plt

#Cv2 documentation: https://docs.opencv.org/3.4/dd/d53/tutorial_py_depthmap.html
#Youtube explanation video: https://youtu.be/hUVyDabn1Mg

#depth = baseline (in mm) * focal length (in mm)/ disparity (in pixels)
"""
The focal length of both cameras have to be the same
"""
baseline = 0 #abs distance between two cameras (Changed based on scenerio)
focal_length = 0 #The focal length of cameras (Changed based on scenerio)



"""
All the sources (images or captures), have to be the same size, file_type, and filename
"""
#Get image by camera (source method 1)
"""
camera_pL = None
camera_pR = None
capL = cv2.VideoCapture(camera_pL)
capR = cv2.VideoCapture(camera_pR)
_, frame_L = capL.read()
_, frame_R = capR.read()
cv2.imwrite("left_img.png", frame_L)
cv2.imwrite("right_img.png", frame_R)
"""



# read left and right images (source method 2)
# using example image right here
imgR = cv2.imread('right_img.png', 0)
print(imgR.shape[:2])
imgR2 = imgR[0:3000, 0:3976] #convert images to same size
print(imgR2.shape[:2])
imgL = cv2.imread('left_img.png', 0)
print(imgL.shape[:2])






# creates StereoBm object
stereo = cv2.StereoBM_create(numDisparities = 16,
							blockSize = 15)

# computes disparity
disparity = stereo.compute(imgL, imgR2)/16 #convert to real floatng point nums by divide the result with 16



# displays image as grayscale and plotted using Matplotlib
plt.imshow(disparity, 'gray')
plt.show()


try:
    depth = baseline * focal_length / disparity
    print("depth =" + depth)
except TypeError:
    print("Invalid value type for either baseling of focal_length, has to be int")