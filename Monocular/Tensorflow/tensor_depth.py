import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
"""
import urllib.request
url, filename = ("https://github.com/intel-isl/MiDaS/releases/download/v2_1/model_opt.tflite", "model_opt.tflite")
urllib.request.urlretrieve(url, filename)
"""


   # load model
interpreter = tf.lite.Interpreter(model_path="/Users/philliao/Documents/Research_Projects/Honeydew_Picker/model_opt.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']



cap = cv2.VideoCapture(1)
while cap.isOpened:
    ret, frame = cap.read()




    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0




    img_resized = tf.image.resize(img, [256,256], method='bicubic', preserve_aspect_ratio=False)
    #img_resized = tf.transpose(img_resized, [2, 0, 1])
    img_input = img_resized.numpy()
    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    img_input = (img_input - mean) / std
    reshape_img = img_input.reshape(1,256,256,3)
    tensor = tf.convert_to_tensor(reshape_img, dtype=tf.float32)

 

    # inference
    interpreter.set_tensor(input_details[0]['index'], tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    output = output.reshape(256, 256)

    # output file
    prediction = cv2.resize(output, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_CUBIC)
    #print(" Write image to: output.png")
    depth_min = prediction.min()
    depth_max = prediction.max()
    img_out = (255 * (prediction - depth_min) / (depth_max - depth_min)).astype("uint8")

    #cv2.imwrite("output.png", img_out)
    plt.imshow(img_out)
    cv2.imshow("CV2", frame)
    plt.pause(0.00001)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()


plt.show()