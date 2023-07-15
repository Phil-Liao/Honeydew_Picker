import cv2
import numpy as np

#Cv2 documentation: https://docs.opencv.org/3.4/dd/d53/tutorial_py_depthmap.html
#Youtube explanation video: https://youtu.be/hUVyDabn1Mg

#depth = baseline (in mm) * focal length (in mm)/ disparity (in pixels)

class image:
    def __init__(self, camera_focal_length:float, img:np.array, img_type:int) -> None:
        self.camera_focal_length = camera_focal_length
        self.img = img
        self.shape = img.shape[:2]
        self.img_type = img_type
    def get_camera_info(self):
        return self.camera_focal_length
    def change_camera_info(self, camera_focal_length:float):
        self.camera_focal_length = camera_focal_length
    def get_img_info(self):
        return [self.img, self.img_type]
    def change_img_info(self, info_type:int, info):
        if info_type == 0:
            self.img = info
        elif info_type == 1:
            self.img_type = info
        else:
            print('[ERROR] Incorrect data type for parameter "info_type", must be int either 0 (img) or 1 (img_type).')
            exit()
    def img_crop(self, width_s:int, width_e:int, height_s:int, height_e:int):
        self.img = self.img[width_s:width_e, height_s:height_e]


def disparity_computing(img_l:image, img_r:image, num_disparities:int, block_size:int):
    if img_l.get_camera_info() != img_r.get_camera_info():
        print('[ERROR] Focal length for "img_l" and "img_r" have to be the same, found to be different.')
        exit()
    elif img_l.get_img_info()[0].shape[0] != img_r.get_img_info()[0].shape[0]:
        print('[ERROR] Image height for "img_l" and "img_r" have to be the same, found to be different.')
        exit()
    elif img_l.get_img_info()[0].shape[1] != img_r.get_img_info()[0].shape[1]:
        print('[ERROR] Image width for "img_l" and "img_r" have to be the same, found to be different.')
        exit()
    elif (img_l.get_img_info()[1] != 0):
        print('[ERROR] Incorrect image type, has to be 0.')
        exit()
    elif (img_r.get_img_info()[1] != 0):
        print('[ERROR] Incorrect image type, has to be 0.')
        exit()
    elif num_disparities < 0:
        print('[ERROR] Value for parameter "num_disparities" has to be int that is greater than or equal to 0.')
    elif (block_size % 2) != 1:
        print('[ERROR] Value for parameter "block_size" has to be int that is an odd number.')
    
    stereo = cv2.StereoBM_create(numDisparities = num_disparities, blockSize = block_size)

    disparity = stereo.compute(img_l.get_img_info()[0], img_r.get_img_info()[0]).astype(np.float32)/16 #convert to real floating point nums by divide the result with 16
    
    possible_disparity_key = []
    possible_disparity_value = []
    for i in disparity:
        for j in i:
            if float(j) not in possible_disparity_key:
                possible_disparity_key.append(float(j))
                possible_disparity_value.append(1)
            else:
                possible_disparity_value[possible_disparity_key.index(float(j))] += 1

    try:
        possible_disparity_value.pop(possible_disparity_key.index(-1.0))
        possible_disparity_key.remove(-1.0)
    except ValueError:
        print('[WEIRD] There is no int -1 as element in list "possible_disparity_key".')
    try:
        possible_disparity_value.pop(possible_disparity_key.index(0.0))
        possible_disparity_key.remove(0.0)
    except ValueError:
        print('[WEIRD] There is no int 0 as element in list "possible_disparity_key".')


    p_d_v_max = max(possible_disparity_value)
    p_d_k_max = possible_disparity_key[possible_disparity_value.index(p_d_v_max)]
    
    return p_d_k_max

def get_actual_depth(baseline:float, focal_length:float, most_possible_disparity:float):
    #baseline in meters (metres)
    #focal_length in meters (metres)
    depth = baseline * focal_length / most_possible_disparity
    return depth




