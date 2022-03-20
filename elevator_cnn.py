#%%
# import libraries
import numpy as np
import pandas as pd
import math
import shutil
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split




#%%
# model creation
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, f1_score
from datetime import datetime

# hyperparameter
N_LAYER = 4


#%%
def cnn(size, n_layers):
  
  # define hyperparameters
  MIN_NEURONS = 20
  MAX_NEURONS = 120
  KERNEL = (3,3)

  # determine the # of neurons in each convolutional layer
  steps = np.floor(MAX_NEURONS/(n_layers+1))
  neurons = np.arange(MIN_NEURONS, MAX_NEURONS, steps)
  nuerons = neurons.astype(np.int32)

  # define model
  model = Sequential()

  # add convolutional layers
  for i in range(0, n_layers):
    if i==0:
      shape = (size[0], size[1], size[2])
      model.add(Conv2D(neurons[i], KERNEL, input_shape=shape))
    else:
      model.add(Conv2D(neurons[i], KERNEL))

    model.add(Activation('relu'))

  # add max pooling layer
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Flatten())
  model.add(Dense(MAX_NEURONS))
  model.add(Activation('relu'))

  # add ouput layer
  model.add(Dense(1))
  model.add(Activation('sigmoid'))

  # compile the model
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

  # print a summary of the model
  model.summary()

  return model


#%%
# copy original elevator data into github directory
labels = ['010', '012', '020', '022', '030', '032', '040', '042', '050', '052', '060', '062',
          '110', '112', '120', '122', '130', '132', '140', '142', '150', '152', '160', '162']
label_num = len(labels)
path_label = ['./elevator_label/'+l+'/' for l in labels]



#%%
# open each label directory as dictionary
label_files = {}
'''
  label_files[key][0] --> cur_csv
  label_files[key][1] --> cur_png
  label_files[key][2] --> act_csv
  label_files[key][3] --> act_png
'''

for i in range(label_num):
  file_list = os.listdir(path_label[i])
  label_files[labels[i]] = []
  csv_list = [file for file in file_list if file.endswith('.csv')]
  png_list = [file for file in file_list if file.endswith('.png')]
  cur_csv_list, cur_png_list = [], []
  act_csv_list, act_png_list = [], []
  for j in range(len(csv_list)):
    if '_active.csv' in csv_list[j]:
      act_csv_list.append(csv_list[j])
      act_png_list.append(png_list[j])
    else:
      cur_csv_list.append(csv_list[j])
      cur_png_list.append(png_list[j])
  label_files[labels[i]].append(cur_csv_list)
  label_files[labels[i]].append(cur_png_list)
  label_files[labels[i]].append(act_csv_list)
  label_files[labels[i]].append(act_png_list)

cnt = 0
for i in range(label_num):
  for j in range(len(label_files[labels[i]][0])):
    cnt += 1
print(cnt)



#%%
# ver1
n_cur_img, n_act_img = cnt, cnt
for i in range(cnt):



#%%
# ver2
# get and save array of pixels from _list dictionary


    import cv2

image_w, image_h = 43, 28
X_cur, X_act, Y_ud, Y_floor = [], [], [], []

for i in range(label_num):
  y_ud = [-1 for x in range(label_num)]
  y_floor = [-1 for x in range(label_num)]
  y_ud[i] = int(labels[i][0])
  y_floor[i] = int(labels[i][1])
  for j in range(len(label_files[labels[i]][1])):
    cimg = cv2.imread(path_label[i]+label_files[labels[i]][1][j], cv2.IMREAD_GRAYSCALE)
    cimg = cimg[36:252, 55:389]
    aimg = cv2.imread(path_label[i]+label_files[labels[i]][3][j], cv2.IMREAD_GRAYSCALE)
    aimg = aimg[36:252, 55:389]
    X_cur.append(cimg)
    X_act.append(aimg)
    Y_ud.append(y_ud)
    Y_floor.append(y_floor)
    #break
  #break
X_cur = np.array(X_cur)
X_act = np.array(X_act)
Y_ud = np.array(Y_ud)
Y_floor = np.array(Y_floor)
#print(Y_ud)



#%%
# ver2
Xcur_train, Xcur_test, Yfloor_train, Yfloor_test = train_test_split(X_cur, Y_floor, test_size=0.2)
Xact_train, Xact_test, Yud_train, Yud_test = train_test_split(X_act, Y_ud, test_size=0.2)
cur_floor_xy = (Xcur_train, Xcur_test, Yfloor_train, Yfloor_test)
act_ud_xy = (Xact_train, Xact_test, Yud_train, Yud_test)

if os.path.isdir('./elevator_test')==False:
  os.mkdir('./elevator_test')

np.save('./elevator_test/cur_floor_xy.npy', cur_floor_xy)
np.save('./elevator_test/act_ud_xy.npy', act_ud_xy)


#%%
# ver2
Xcur_train, Xcur_test, Yfloor_train, Yfloor_test = np.load('./elevator_test/cur_floor_xy.npy', allow_pickle=True)
Xact_train, Xact_test, Yud_train, Yud_test = np.load('./elevator_test/act_ud_xy.npy', allow_pickle=True)

model = Sequential() # 모델 생성
model = Sequential()
model.add(Conv2D(16, 3, 3, padding='same', activation='relu', input_shape=Xcur_train.shape[1:])) # 모델에 레이어 추가
model.add(MaxPooling2D(pool_size=(2, 2)))


#%%
# ver1
EPOCHS = 150
BATCH_SIZE = 200
PATIENCE = 10
early_stopping = EarlyStopping(monitor='loss', min_delta=0, patience=PATIENCE, verbose=0, mode='auto')

# TensorBoard callback
LOG_DIRECTORY_ROOT = ''
now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
log_dir = '{}/run-{}/'.format(LOG_DIRECTORY_ROOT, now)
tensorboard = TensorBoard(log_dir=log_dir, write_graph=True, write_images=True) # tensorboard 실시간 시각화

# Place the callbacks in a list
callbacks = [early_stopping, tensorboard]

#%%
# Train the model
model.fit()
