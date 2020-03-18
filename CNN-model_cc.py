# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 20:08:36 2020

@author: alexr
"""


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, MaxPool2D, Conv2D, Dropout
from sklearn.model_selection import train_test_split
import pickle
from keras.utils import to_categorical


X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

Y = []
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

for i in y:
  try:
    Y.append(genres.index(i))
  except ValueError:
    Y.append(genres.index(i[:-4]))
    
Y = to_categorical(Y)
model1 = Sequential()

model1.add(Conv2D(32, (3, 3), input_shape = X.shape[1:]))
model1.add(Activation("relu"))
model1.add(MaxPool2D(pool_size = (2, 2)))


model1.add(Conv2D(16, (3, 3)))
model1.add(Activation('relu'))
model1.add(MaxPool2D(pool_size =(2, 2)))

model1.add(Conv2D(8, (3, 3)))
model1.add(Activation('relu'))
model1.add(MaxPool2D(pool_size =(2, 2)))
'''
model1.add(Conv2D(8, (3, 3)))
model1.add(Activation('relu'))
model1.add(MaxPool2D(pool_size =(2, 2)))
'''
model1.add(Flatten())
#model1.add(Dense(8))

model1.add(Dense(10))
model1.add(Activation("softmax"))
model1.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])

mses = []

for i in range(0, 15):
  X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
  model1.fit(X_train, y_train, batch_size = 5, epochs = 5)
predictions = model1.predict(X_test)
print(model1.evaluate(X_test, y_test, verbose = 2))
