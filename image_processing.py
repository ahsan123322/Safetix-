import cv2
import numpy as np
from keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing.image import img_to_array

def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (160, 160))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    return image