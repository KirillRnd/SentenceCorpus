# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 10:50:26 2019

@author: Кирилл
"""

import numpy as np

from itertools import product
import functools
from keras import backend as K
import tensorflow as tf
def createVecFromString(string,dictionary,size_t=30):
    arr=np.zeros(shape=(size_t,4))
    Li_words=string.split(' ')
    Li_words = list(filter(None, Li_words))
    #print(Li_words)
    arr_tmp=np.array(dictionary.GetRndVec(Li_words,create=False,ret=True))
    
    if len(arr_tmp)-len(arr)>0:
        print('Wrong size_t. Increase it')
    
    arr_start=np.random.randint(np.abs(len(arr_tmp)-len(arr)))
    #print(len(arr_tmp))
    arr[arr_start:arr_start+len(arr_tmp)]=arr_tmp
    return arr
def createVecFromStringDefault(dictionary,size_t=30):
    def TmpFunc(string):
        return createVecFromString(string,dictionary,size_t)
    return TmpFunc

def as_keras_metric(method):
    
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper
def w_categorical_crossentropy(y_true, y_pred, weights):
    nb_cl = len(weights)
    final_mask = K.zeros_like(y_pred[:, 0])
    y_pred_max = K.max(y_pred, axis=1)
    y_pred_max = K.expand_dims(y_pred_max, 1)
    y_pred_max_mat = K.equal(y_pred, y_pred_max)
    for c_p, c_t in product(range(nb_cl), range(nb_cl)):

        final_mask += (K.cast(weights[c_t, c_p],K.floatx()) * K.cast(y_pred_max_mat[:, c_p] ,K.floatx())* K.cast(y_true[:, c_t],K.floatx()))
    return K.categorical_crossentropy(y_pred, y_true) * final_mask

def create_w_matrix(weights):
    arr=np.ones((len(weights),len(weights)))
    for i,v_1 in enumerate(weights):
        for j,v_2 in enumerate(weights):
            arr[i,j]=weights[i]/weights[j]
    return arr