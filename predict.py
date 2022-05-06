import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib as plt
  


filepath = 'models\him_group4_model.h5'    # File path
model = tf.keras.models.load_model(filepath, compile = True)
image = np.array(Image.open("dummyimages\partialmask.jpg").resize((160, 160)))
images_list = []
images_list.append(np.array(image))
x = np.asarray(images_list)

clas = int(np.argmax(model.predict(x), axis=-1))


class_names = ["partial mmask","withmask","withoutmask"]

print(class_names[clas])

