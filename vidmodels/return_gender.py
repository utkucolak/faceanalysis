#!/usr/bin/env python
# coding: utf-8


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.image import imread
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
 
import logging
tf.get_logger().setLevel(logging.ERROR)
os.chdir('..')

data_path = 'C:\\Users\\Casper\\Desktop\\faceanalysis\\datasets\\face_gender\\'
def create_data(data_path):

    train_gen = ImageDataGenerator(rescale=1/255, rotation_range=0.2, zoom_range=0.2, horizontal_flip=True)
    train_data = train_gen.flow_from_directory(data_path+'train', target_size=(261,195), batch_size=16, color_mode='rgb', class_mode='binary')
    test_gen = ImageDataGenerator(rescale=1/255)
    test_data = test_gen.flow_from_directory(data_path+'test', target_size=(261,195), batch_size=16, color_mode='rgb', class_mode='binary', shuffle=False)
    class_names = train_data.class_indices
    return (train_data, test_data, class_names)
def create_model():
    
    model = Sequential()
    model.add(Conv2D(32, (3,3), input_shape=(261,195,3), activation='relu'))
    model.add(MaxPooling2D(2,2))
    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(MaxPooling2D(2,2))
    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(MaxPooling2D(2,2))
    model.add(Conv2D(32, (3,3), activation='relu'))
    model.add(MaxPooling2D(2,2))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model


def train_model(data_path):
    train_data, test_data, _ = create_data(data_path)
    model = create_model()
    
    model.fit_generator(train_data, validation_data=test_data, epochs=50)

    model.save_weights('my_gender_weights')
    model.save('my_gender_model')
def return_gender(frame):
    model = create_model()
    model.load_weights('C:\\Users\\Casper\\Desktop\\faceanalysis\\my_gender_weights')
    
    frame = tf.expand_dims(frame,0)
    p = model.predict(frame)
    liste = []
    i,j,class_names = create_data(data_path)
    for x,y in class_names.items():
        liste.append(x)
    return(liste[int(np.round(p[0][0]))])

