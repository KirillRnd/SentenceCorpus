# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 11:02:01 2019

@author: Кирилл
"""

from AdditionalF import as_keras_metric
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
from keras.layers import Dropout, Flatten, Bidirectional,LSTM
#from sklearn.utils import class_weight
#from keras.optimizers import SGD
#from itertools import product
#from functools import partial
import pickle
def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)
data_4_y_t=load_obj('data_4_y_t')
data_4_y_t_cat=load_obj('data_4_y_t_cat')
data_4_y=load_obj('data_4_y')
data_4_y_cat=load_obj('data_4_y_cat')
data_4_X_t=load_obj('data_4_X_t')
data_4_X=load_obj('data_4_X')
auc_roc = as_keras_metric(tf.metrics.auc)
L2 = 1e-5
DROPOUT = 0.4
RDROPOUT = 0.2
from sklearn.utils import class_weight
my_class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(data_4_y),
                                                 data_4_y) 
def create_baseline_dense():
    model = Sequential()
    #model.add(Dropout(0.20))
    #model.add(LSTM(52,input_shape=(50,32), return_sequences=True))
    model.add(Bidirectional(LSTM(
            50, kernel_regularizer=regularizers.l2(L2),
            recurrent_regularizer=regularizers.l2(L2),
            bias_regularizer=regularizers.l2(L2), dropout=DROPOUT,
            recurrent_dropout=RDROPOUT,
            activation='tanh', return_sequences=True)))
    model.add(Bidirectional(LSTM(
            50, kernel_regularizer=regularizers.l2(L2),
            recurrent_regularizer=regularizers.l2(L2),
            bias_regularizer=regularizers.l2(L2), dropout=DROPOUT,
            recurrent_dropout=RDROPOUT,
            return_sequences=True)))
    model.add(Bidirectional(LSTM(
            50, kernel_regularizer=regularizers.l2(L2),
            recurrent_regularizer=regularizers.l2(L2), dropout=DROPOUT,
            recurrent_dropout=RDROPOUT,
            return_sequences=True)))
    model.add(Flatten())
    model.add(Dense(5, activation='softmax',
                kernel_regularizer=regularizers.l2(L2),
                activity_regularizer=regularizers.l1(L2)))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc',auc_roc])
    return model

dense_m=create_baseline_dense()
print("Запуск модели")
dense_m.fit(data_4_X, data_4_y_cat,validation_data=(data_4_X_t, data_4_y_t_cat), epochs=100, batch_size=64,class_weight=my_class_weights)

from sklearn.metrics import confusion_matrix
y_v=dense_m.predict_classes(data_4_X_t)
print(confusion_matrix(data_4_y_t, y_v))
dense_m.save('mod.hdf5')