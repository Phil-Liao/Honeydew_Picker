# import OpenCV and pyplot
import cv2
import numpy as np
from matplotlib import pyplot as plt

#Cv2 documentation: https://docs.opencv.org/3.4/dd/d53/tutorial_py_depthmap.html
#Youtube explanation video: https://youtu.be/hUVyDabn1Mg

#depth (in cm) = baseline (in mm) * focal length (in px)/ disparity (in pixels)
"""
The focal length of both cameras have to be the same
"""
baseline = 0.06 #abs distance between two cameras(Changed based on scenerio, unit in "meters")
focal_length = 25 #The focal length of cameras(Changed based on scenerio, unit in "pixels")
actual_distance = None #The actual distance of the object(final result, unit in "meters")


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
imgR = cv2.imread('/Users/philliao/Documents/Research_Projects/Honeydew_Picker/Assets/right_img.png', 0)
#print(imgR.shape[:2])
imgR2 = imgR[0:3000, 0:3976] #convert images to same size
#print(imgR2.shape[:2])
imgL = cv2.imread('/Users/philliao/Documents/Research_Projects/Honeydew_Picker/Assets/left_img.png', 0)
#print(imgL.shape[:2])
#print(imgL.shape == imgR.shape)





# creates StereoBm object
stereo = cv2.StereoBM_create(numDisparities = 16,
							blockSize = 15)

# computes disparity
disparity = stereo.compute(imgL, imgR2).astype(np.float32)/16 #convert to real floatng point nums by divide the result with 16




# displays image as grayscale and plotted using Matplotlib
plt.imshow(disparity, 'gray')
plt.show()






possible_disparity_key = []
possible_disparity_value = []
for i in disparity:
    for j in i:
        if float(j) not in possible_disparity_key:
            possible_disparity_key.append(float(j))
            possible_disparity_value.append(1)
        else:
            possible_disparity_value[possible_disparity_key.index(float(j))] += 1




print(possible_disparity_key.index(-1.0))
print(possible_disparity_key.index(0.0))
possible_disparity_value.pop(possible_disparity_key.index(-1.0))
possible_disparity_key.remove(-1.0)
possible_disparity_value.pop(possible_disparity_key.index(0.0))
possible_disparity_key.remove(0.0)



print("possible_disparity_key = " + str(possible_disparity_key))
print("possible_disparity_value = " + str(possible_disparity_value))


p_d_v_max = max(possible_disparity_value)
p_d_k_max = possible_disparity_key[possible_disparity_value.index(p_d_v_max)]

print("p_d_v_max = " + str(p_d_v_max))
print("p_d_k_max = " + str(p_d_k_max))







depth = baseline * focal_length / p_d_k_max

print(depth)
