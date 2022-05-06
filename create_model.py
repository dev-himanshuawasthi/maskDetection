import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def main(batch_size,img_height,img_width,seed,epochs):


  # load images from directory 

  TRAINING_DIR = r'C:\Users\himaw\Desktop\deeplearning\imagedata\MP2_FaceMask_Dataset\test'
  VALIDATION_DIR = r'C:\Users\himaw\Desktop\deeplearning\imagedata\MP2_FaceMask_Dataset\train'

  train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAINING_DIR,
    seed=seed,
    image_size=(img_height, img_width),
    batch_size=batch_size
    )

  val_ds = tf.keras.utils.image_dataset_from_directory(
    VALIDATION_DIR,
    seed=seed,
    image_size=(img_height, img_width),
    batch_size=batch_size
    )


  # classes present in the data
  class_names = train_ds.class_names
  print(class_names)


  for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break


  # normalize images
  normalization_layer = tf.keras.layers.Rescaling(1./255)

  normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
  image_batch, labels_batch = next(iter(normalized_ds))
  first_image = image_batch[0]
  # Notice the pixel values are now in `[0,1]`.
  print(np.min(first_image), np.max(first_image))



  AUTOTUNE = tf.data.AUTOTUNE

  train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
  val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


  # define model
  num_classes = 3

  model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='softmax'),
    tf.keras.layers.Dense(num_classes)
  ])



  # compile model
  model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])


  # fit network

  model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
  )

  # model summary 
  model.summary()


  # save model to directory
  model.save("models\c_model.h5")
  print("Saved model to disk")


if __name__== '__main__':
    
    BATCH_SIZE=32
    IMG_HEIGHT=160
    IMG_WIDTH=160
    SEED=123
    EPOCHS= 1
    main(batch_size=BATCH_SIZE,img_height=IMG_HEIGHT,img_width=IMG_WIDTH,seed=SEED,epochs=EPOCHS)


