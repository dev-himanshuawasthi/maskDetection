import cv2
import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib as plt
  
  
# File path
filepath = 'models\himanshu_model.h5'

# Load the model
model = tf.keras.models.load_model(filepath, compile = True)

image = np.array(Image.open("51014_Mask_Mouth_Chin.jpg").resize((160, 160)))
images_list = []
images_list.append(np.array(image))
x = np.asarray(images_list)

pr_mask = model.predict(x)
print(pr_mask)



