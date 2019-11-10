import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean
filepath='melanoma_imagepool\\'
# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details)
print(output_details)
imgs = [mpimg.imread('ISIC_0000121.jpg'),mpimg.imread('ISIC_0000154.jpg')]
for imgs in os.listdir(filepath):
    img = mpimg.imread(filepath+'\\'+imgs)
    image_resized = resize(img, [1, 224, 224, 3])
#np.reshape(img, [1, 224, 224, 3], order='C')
    print(image_resized.shape)


# Test model on random input data.

    input_data = np.array(image_resized, dtype=np.uint8)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()
    print(interpreter.get_output_details())
# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    output = interpreter.tensor(interpreter.get_output_details()[0]["index"])
    print(output_data)
    print("inference %s" % output())
