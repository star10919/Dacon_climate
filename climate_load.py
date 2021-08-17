import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import Adam

from datetime import datetime
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import numpy as np
import pandas as pd

import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')

from glob import glob

def get_input_dataset(data, index, train = False) : 
    input0 = tf.convert_to_tensor(data[0][index].toarray(), tf.float32)
    input1 = tf.convert_to_tensor(data[1][index].toarray(), tf.float32)
    input2 = tf.convert_to_tensor(data[2][index].toarray(), tf.float32)
    
    if train : 
        label = labels[index]

        return input0, input1, input2, label
    else:
        return input0, input1, input2,

def single_dense(x, units):
    fc = Dense(units, activation = None, kernel_initializer = 'he_normal')(x)
    batch = BatchNormalization()(fc)
    relu = ReLU()(batch)
    dr = Dropout(0.2)(relu)
    
    return dr

def create_model(input_shape0,input_shape1,input_shape2, num_labels, learning_rate):
    x_in0 = Input(input_shape0,)
    x_in1 = Input(input_shape1,)
    x_in2 = Input(input_shape2,)
    
    fc0 = single_dense(x_in0, 512)  #512
    # fc0 = single_dense(fc0, 256)
    fc0 = single_dense(fc0, 100)    #128
    fc0 = single_dense(fc0, 64)
    
    fc1 = single_dense(x_in1, 512)  # 1024
    # fc1 = single_dense(fc1, 256)
    fc1 = single_dense(fc1, 128)    #128
    fc1 = single_dense(fc1, 64)
    
    fc2 = single_dense(x_in2, 512)
    # fc2 = single_dense(fc2, 256)
    fc2 = single_dense(fc2, 100)    #128
    fc2 = single_dense(fc2, 64)
    
    fc = Concatenate()([fc0,fc1,fc2])
    # fc = single_dense(fc, 128)
    fc = single_dense(fc, 64)   #64
    
    x_out = Dense(num_labels, activation = 'softmax')(fc)
    
    model = Model([x_in0,x_in1,x_in2], x_out)
    optimizer = Adam(lr=learning_rate)
    model.compile(optimizer = optimizer, loss = 'sparse_categorical_crossentropy', metrics = ['acc'])
    
    return model

###################### load
with open('./_save/_npy/climate_ccc_savemodel_3.pkl','rb') as f :       # ^^
    train_inputs, test_inputs, labels = pickle.load(f)

num_labels = 46
learning_rate = 0.04        #0.05
seed = np.random.randint(2**16-1)
skf = StratifiedKFold(n_splits=5, shuffle = True, random_state = seed)

for train_idx, valid_idx in skf.split(train_inputs[0], labels):
    X_train_input0, X_train_input1, X_train_input2, X_train_label = get_input_dataset(train_inputs, train_idx, train = True)
    X_valid_input0, X_valid_input1, X_valid_input2, X_valid_label = get_input_dataset(train_inputs, valid_idx, train = True)
    
    now = datetime.now()
    now = str(now)[11:16].replace(':','h')+'m'
    ckpt_path = f'./_save/mcp/dacon_climatetech_{now}.ckpt'

    input_shape0 = X_train_input0.shape[1]
    input_shape1 = X_train_input1.shape[1]
    input_shape2 = X_train_input2.shape[1]


    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5),   # 5
        tf.keras.callbacks.ModelCheckpoint(ckpt_path, monitor = 'val_acc', save_best_only= True, save_weights_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor = 'val_loss', factor=0.7, patience = 3),  #0.7, 3
                ]
    model = create_model(input_shape0,input_shape1,input_shape2, num_labels, learning_rate)
    model.fit(
                        [X_train_input0,X_train_input1,X_train_input2],
                        X_train_label,
                        epochs=1000,
                        callbacks=callbacks,
                        validation_data=([X_valid_input0, X_valid_input1, X_valid_input2], X_valid_label),
                        verbose=1,  # Logs once per epoch.
                        batch_size=1180)    #1200
    
    # model.load_weights(ckpt_path)
    prediction = model.predict([test_inputs[0], test_inputs[1], test_inputs[2]])
    np.save(f'./_save/mcp/climatetech_grouping{now}_prediction.npy', prediction)

predictions = []
for ar in glob('./_save/mcp/climatetech_grouping*.npy'):
    arr = np.load(ar)
    predictions.append(arr)

sample = pd.read_csv('./project_climate(dacon)/_data/sample_submission.csv')
sample['label'] = np.argmax(np.mean(predictions,axis=0), axis = 1)
sample.to_csv('./project_climate(dacon)/Cli_cccccc_26.csv', index=False)   #바꾸기



with open('./_save/_npy/climate_ccc_savemodel_3.pkl','rb') as f :   # ^^
    train_inputs, test_inputs, labels = pickle.load(f)

'''
ccccc_7  save3       <sub>
Epoch 18/1000
110/110 [==============================] - 10s 90ms/step - loss: 0.1053 - acc: 0.9685 - val_loss: 0.3252 - val_acc: 0.9350

ccccc_8-save3        <sub>
Epoch 16/1000
110/110 [==============================] - 10s 91ms/step - loss: 0.1308 - acc: 0.9599 - val_loss: 0.3310 - val_acc: 0.9403

ccccc_12 save3        <sub>
Epoch 14/1000
110/110 [==============================] - 9s 82ms/step - loss: 0.0938 - acc: 0.9702 - val_loss: 0.3588 - val_acc: 0.9384

ccccc_13 save3        <sub>
Epoch 15/1000
110/110 [==============================] - 8s 70ms/step - loss: 0.0922 - acc: 0.9695 - val_loss: 0.3177 - val_acc: 0.9425

ccccc_14 save3        <sub>
Epoch 19/1000
110/110 [==============================] - 8s 71ms/step - loss: 0.0883 - acc: 0.9734 - val_loss: 0.3056 - val_acc: 0.9378

ccccc_15 save3      홍 <sub>
Epoch 17/1000
114/114 [==============================] - 8s 68ms/step - loss: 0.0824 - acc: 0.9726 - val_loss: 0.3412 - val_acc: 0.9390

ccccc_16 save5      홍 <sub>
Epoch 15/1000
116/116 [==============================] - 9s 79ms/step - loss: 0.0914 - acc: 0.9705 - val_loss: 0.3527 - val_acc: 0.9416

ccccc_17 save3      홍 <sub>
Epoch 15/1000
112/112 [==============================] - 9s 76ms/step - loss: 0.0817 - acc: 0.9731 - val_loss: 0.3394 - val_acc: 0.9409

ccccc_18 save3        <sub>
Epoch 14/1000
112/112 [==============================] - 8s 71ms/step - loss: 0.1054 - acc: 0.9652 - val_loss: 0.3309 - val_acc: 0.9384

ccccc_19 save3   *****  <sub>
Epoch 13/1000
112/112 [==============================] - 9s 77ms/step - loss: 0.1171 - acc: 0.9618 - val_loss: 0.3158 - val_acc: 0.9359

ccccc_20 save3
Epoch 15/1000
112/112 [==============================] - 8s 67ms/step - loss: 0.0827 - acc: 0.9723 - val_loss: 0.3452 - val_acc: 0.9412

ccccc_21 save3
Epoch 12/1000
112/112 [==============================] - 8s 70ms/step - loss: 0.0941 - acc: 0.9690 - val_loss: 0.3325 - val_acc: 0.9402

ccccc_22 save3
Epoch 11/1000
112/112 [==============================] - 8s 70ms/step - loss: 0.0946 - acc: 0.9691 - val_loss: 0.3140 - val_acc: 0.9421

ccccc_23 save3
Epoch 11/1000
111/111 [==============================] - 8s 68ms/step - loss: 0.1039 - acc: 0.9658 - val_loss: 0.3439 - val_acc: 0.9377

ccccc_24 save3
Epoch 13/1000
112/112 [==============================] - 8s 68ms/step - loss: 0.0769 - acc: 0.9751 - val_loss: 0.3348 - val_acc: 0.9427

ccccc_25 save3
Epoch 17/1000
112/112 [==============================] - 8s 70ms/step - loss: 0.0776 - acc: 0.9753 - val_loss: 0.3453 - val_acc: 0.9409

ccccc_26 save3
Epoch 16/1000
112/112 [==============================] - 8s 69ms/step - loss: 0.0855 - acc: 0.9719 - val_loss: 0.3474 - val_acc: 0.9423



'''