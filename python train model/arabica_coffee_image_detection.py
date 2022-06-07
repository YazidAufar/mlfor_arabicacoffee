# -*- coding: utf-8 -*-
"""Arabica Coffee - Image Detection.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1dzpoearJSX-mDXRnrAgFQCIXtgXbos_D
"""

from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import  load_img, ImageDataGenerator
from tensorflow.keras.layers import Dense, Conv2D, GlobalAvgPool2D, Input
from tensorflow.keras import callbacks, optimizers
import tensorflow as tf

from google.colab import drive 
import numpy as np
import os
import matplotlib.pyplot as plt ### plotting bar chart

drive.mount('/content/drive')

# Commented out IPython magic to ensure Python compatibility.
# %cd drive/MyDrive/Coffee

!pwd

!unzip archive.zip

# from translate import translate

# translate

# for i in os.listdir("raw-img"):
#   try:
#     os.rename("raw-img/"+i,"raw-img/"+translate[i])
#   except Exception as e:
#     print(e)

!ls raw-img

# for i in os.listdir("raw-img"):
#   print(i,len(os.listdir("raw-img/"+i)))

# for i in os.listdir("raw-img"):
#   print (i)

for i in os.listdir("raw-img"):
  print(i,len(os.listdir("raw-img/"+i)))

# try:
#   os.mkdir("train")
#   os.mkdir("test")
# except:
#   pass
# for i in os.listdir("raw-img"):
#   try:
#     os.mkdir("train/"+i)
#     os.mkdir("test/"+i)
#   except:
#     pass
#   for j in os.listdir("raw-img/"+i)[:1000]:
#     os.rename("raw-img/"+i+"/"+j, "train/"+i+"/"+j)
#   for j in os.listdir("raw-img/"+i)[:400]:
#     os.rename("raw-img/"+i+"/"+j, "test/"+i+"/"+j)

try:
  os.mkdir("train")
  os.mkdir("test")
except:
  pass
for i in os.listdir("raw-img"):
  try:
    os.mkdir("train/"+i)
    os.mkdir("test/"+i)
  except:
    pass
  for j in os.listdir("raw-img/"+i)[:5200]:
    os.rename("raw-img/"+i+"/"+j, "train/"+i+"/"+j)
  for j in os.listdir("raw-img/"+i)[:1300]:
    os.rename("raw-img/"+i+"/"+j, "test/"+i+"/"+j)

!ls train

!ls test

def img_data(dir_path, target_size, batch, class_lst, preprocessing):
  if preprocessing:
    gen_object = ImageDataGenerator(preprocessing_function=preprocessing)
  else:
    gen_object = ImageDataGenerator()

  return(gen_object.flow_from_directory(dir_path, 
                                   target_size=target_size, 
                                   batch_size=batch, 
                                   class_mode='sparse', 
                                   classes=class_lst,
                                   shuffle=True))

train_data_generator = img_data('train', (224,224), 500, os.listdir('train'), preprocess_input)
valid_data_generator = img_data('test',  (224,224), 500, os.listdir('test'),  preprocess_input)

train_data_generator

base_model = tf.keras.applications.mobilenet_v2.MobileNetV2(
            input_shape=(224,224,3), alpha=1.0, include_top=False, weights='imagenet',
            input_tensor=None, pooling=None, classes=1000,
            classifier_activation='softmax')

base_model.trainable = False

base_model.summary()

model = tf.keras.models.Sequential()
model.add(base_model)
model.add(GlobalAvgPool2D())
model.add(Dense(1024, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

early_stop = callbacks.EarlyStopping(monitor='loss', patience=5, mode='min')
save_checkpoint = callbacks.ModelCheckpoint('.coffee_model_new.hdf5', save_best_only=True, monitor='loss', mode='min')

history = model.fit(train_data_generator, batch_size=100, validation_data=valid_data_generator, callbacks=[early_stop, save_checkpoint], epochs=1)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train_loss', 'val_loss'])
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train_accuracy', 'val_accuracy'])
plt.show()

history.history

train_data_generator.class_indices

model.save('coffe_model_new.hdf5')