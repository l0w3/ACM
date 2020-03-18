# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 17:55:11 2020

@author: alexr
"""


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, MaxPool2D, Conv2D, Dropout
from sklearn.model_selection import train_test_split
import pickle


X = pickle.load(open("X.pickle", "rb"))
y = pickle.load(open("y.pickle", "rb"))

Y = []
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

for i in y:
  try:
    Y.append(genres.index(i))
  except ValueError:
    Y.append(genres.index(i[:-4]))
    
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size = (2, 2)))


model.add(Conv2D(16, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size =(2, 2)))

model.add(Conv2D(8, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size =(2, 2)))

model.add(Conv2D(8, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size =(2, 2)))

model.add(Flatten())
model.add(Dense(8))

model.add(Dense(10))
model.add(Activation("softmax"))
model.compile(loss = "sparse_categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])

mses = []

for i in range(0, 10):
  X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)
  model.fit(X_train, y_train, batch_size = 5, epochs = 5)
predictions = model.predict(X_test)
print(model.evaluate(X_test, y_test, verbose = 2))