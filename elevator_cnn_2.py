#%%
# import libraries
import numpy as np
import pandas as pd
import math
import shutil
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

%matplotlib inline


#%%
# model creation
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.models import load_model
from datetime import datetime



#%%
import cv2
# Get and save array of pixels without scaling
path = './elevator_label/'
labels = ['010', '012', '020', '022', '030', '032', '040', '042', '050', '052', '060', '062',
          '110', '112', '120', '122', '130', '132', '140', '142', '150', '152', '160', '162']
label_num = len(labels)

#image_w, image_h = 43, 28
X_act, Y_up = [], [] # active power --> up/down

for idx, l in enumerate(labels):
  updown = [-1 for i in range(label_num)]
  updown[idx] = int(l[0])
  img_dir = path+l+'/'
  for top, dir, f in os.walk(img_dir):
    for filename in f:
      if not '_active.png' in filename: # filter only '_active.png'
        continue
      #aimg = cv2.imread(img_dir+filename, cv2.IMREAD_GRAYSCALE)
      aimg = cv2.imread(img_dir+filename)
      aimg = aimg[36:252, 54:389] # valid area without axis
      X_act.append(aimg)
      Y_up.append(updown)

X_act = np.array(X_act)
Y_up = np.array(Y_up)
print(X_act.shape)

Xact_train, Xact_test, Yup_train, Yup_test = train_test_split(X_act, Y_up)
act_up_xy = (Xact_train, Xact_test, Yup_train, Yup_test)

np.save('./elevator_test/act_up_xy.npy', act_up_xy)





#%%
# Create model
Xact_train, Xact_test, Yup_train, Yup_test = np.load('./elevator_test/act_up_xy.npy', allow_pickle=True)

model = Sequential()
model = Sequential()
model.add(Conv2D(16, 3, 3, padding='same', activation='relu', input_shape=Xact_train.shape[1:]))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

#%%
