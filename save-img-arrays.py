# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 21:28:15 2020

@author: alexr
"""


import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle
import numpy as np

genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

training_data = []

for g in genres:
  for filename in os.listdir(f'D:\AMC\img_data\{g}'):
    image = cv2.imread(f'D:\AMC\img_data\{g}\{filename}')
    a = filename
    lbl = a[:-9]
    training_data.append([image, lbl])

random.shuffle(training_data)

X = []
y = []
for features, lbls in training_data:
  X.append(features)
  y.append(lbls)
X = np.array(X)


save_X = open('X.pickle', 'wb')
pickle.dump(X, save_X)
save_X.close()
save_y = open('y.pickle', 'wb')
pickle.dump(y, save_y)
save_y.close()
