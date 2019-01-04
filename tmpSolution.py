
# coding: utf-8
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split


import re
import os 
import Levenshtein as lv
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers

directory = './SentenceCorpus/labeled_articles' 
files = os.listdir(directory) 
texts = filter(lambda x: x.endswith('.txt'), files) 
b=list(texts)

data_1=list()
for elem in b:
    f = open(directory+'/'+elem, 'r')
    lines = f.readlines()
    data_1.extend(lines)
    f.close()
def annotation(x):
    if x.find('###')==-1:
        return 1
    else:
        return 0
data_2 = filter(annotation, data_1)
data_2 = list(data_2)

for counter, value in enumerate(data_2):
    data_2[counter]=re.sub(r'[^a-zA-Z ]','',value.replace('\n','').replace('\t','').lower())

directory = './SentenceCorpus/word_lists' 
files = os.listdir(directory) 
words = filter(lambda x: x.endswith('.txt'), files) 
words=list(words)


def getwords(wordlist):
    f = open(directory+'/'+wordlist, 'r')
    lines = f.readlines()
    f.close()
    for counter, value in enumerate(lines):
        lines[counter]=value.replace('\n','').lower()
    return lines



s = np.random.seed(42)
#Мой самописный класс для создания словаря с 4-векторами для каждого слова.
#Слова близкие по Левенштейну(+опечатки) находятся ближе друг к другу в 4-пространстве
class WordList:
    def __init__(self):
        self.dict={}
    def Push(self, words, vector):
        if type(words)==list:
            for elem in words:
                self.dict.update( {elem : vector} )
        if type(words)==str:
            self.dict.update( {words : vector} )
    def GetKeys(self):
        return list(self.dict.keys())
    def GetVal(self,key):
        return self.dict.get(key)
    def FindClosest(self,word):
        closest=''
        dist=100
        for elem in self.GetKeys():
            dist_this=lv.distance(elem,word)
            if dist_this<dist :
                closest=elem
                dist=dist_this
        return {'dist':dist,'cls':closest}
    def FindVec(self,word):
        d=self.FindClosest(word)
        #print(d)
        if d.get('cls')=='':
            return np.array([0,0,0,0])
        else:
            #print(self.GetVal(word))
            return self.GetVal(d.get('cls'))
        return 0
    def RndVec(self):
        r = np.random.normal
        s=1
        d=4
        inv_d = 1.0 / d
        gauss=r(0,s,size=4)
        length = np.linalg.norm(gauss)
        if length == 0.0:
            v = gauss
        else:
            r = np.random.rand() ** inv_d
            v = np.multiply(gauss, r / length)
        return v
    def GetRndVec(self,words,create=False,ret=True):#словарь строится за n!/k! операций, где n - кол-во слов. Переделать
        def GetOne(word,create=False,ret=True):
            d=self.FindClosest(word)
            p=np.exp(-d.get('dist'))
            #print(p)
            q=1-p
            v=self.RndVec()*q+self.FindVec(word)*p
            if (create == True) and (d.get('dist')!=0):
                self.Push(word, v)
            if ret:
                return v
        if ret:
            L=[]
        if type(words)==list:
            for elem in words:
                x=GetOne(elem,create,ret)
                if ret:
                    L.append(x)
        if type(words)==str:
            x=GetOne(words,create,ret)
            if ret:
                    L.append(x)
        if ret:
            return L


aim=getwords(words[0])
base=getwords(words[1])
contrast=getwords(words[2])
own=getwords(words[3])
stopwords=getwords(words[4])


ww=WordList()

ww.Push(aim,     np.array([1,0,0,0]))
ww.Push(base,    np.array([0,1,0,0]))
ww.Push(contrast,np.array([0,0,1,0]))
ww.Push(own,     np.array([0,0,0,1]))

ww.Push(stopwords,np.array([0,0,0,0]))
print("Словарь создан")


data_3=list()
for st in data_2:
    tmpstr1=st[0:4]
    tmpstr2=st[4:-1].strip()
    data_3.append([tmpstr1,tmpstr2])


data_str=pd.DataFrame(data_3)
data_str[1][2]
for row in data_str[1]:
    ww.GetRndVec(row.split(' '),create=True,ret=False)
print("Словарь заполнен")

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

data_4=list()
#classes={
#    'misc':np.array([1,0,0,0,0]),
#    'aimx':np.array([0,1,0,0,0]),
#    'ownx':np.array([0,0,1,0,0]),
#   'cont':np.array([0,0,0,1,0]),
#    'base':np.array([0,0,0,0,1]),
#}

#for row in data_3:
#    label=row[0]
#    para=row[1]
#    data_4.append([classes.get(label),F(para)])


#arr=np.zeros(shape=(30,4))



#data_vec=pd.DataFrame(data_4)


#data_vec.head()

#data_vec.to_csv('Vecoric.csv', sep='\t')




X=data_str[0]
y=data_str[1]


TRAIN_SIZE = 0.7
X_train, X_test, y_train, y_test = train_test_split(y, X, train_size=TRAIN_SIZE, random_state=42)

print("Данные готовы")



# In[97]:



classes={
    'misc':0,
    'aimx':1,
    'ownx':2,
    'cont':3,
    'base':4,
}
sizeOfSet=500
IDs_train=list(np.random.randint(len(X_train),size=sizeOfSet))
data_4_X=np.zeros((sizeOfSet,100,4))
data_4_y=np.zeros((sizeOfSet))
y_train=list(y_train)
X_train=list(X_train)
F=createVecFromStringDefault(ww,size_t=100)
i=0
for Id in IDs_train:
    label=y_train[Id]
    para=X_train[Id]
    data_4_X[i]=F(para)
    data_4_y[i]=classes.get(label)
    print(i)
    i=i+1
    
data_4_y_cat=keras.utils.to_categorical(data_4_y, num_classes=5)
# In[98]:
sizeOfSet_test=200
IDs_test=list(np.random.randint(len(X_test),size=sizeOfSet_test))
data_4_X_t=np.zeros((sizeOfSet_test,100,4))
data_4_y_t=np.zeros((sizeOfSet_test))
y_test=list(y_test)
X_test=list(X_test)
F=createVecFromStringDefault(ww,size_t=100)
i=0
for Id in IDs_test:
    label=y_test[Id]
    para=X_test[Id]
    data_4_X_t[i]=F(para)
    data_4_y_t[i]=classes.get(label)
    print(i)
    i=i+1
data_4_y_t_cat=keras.utils.to_categorical(data_4_y_t, num_classes=5)    
# In[98]
import functools
from keras import backend as K
import tensorflow as tf
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
# In[98]
from keras.layers import Dropout, Flatten
from sklearn.utils import class_weight
from keras.optimizers import SGD
from itertools import product
from functools import partial
my_class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train),
                                                 y_train) 
w_matrix=create_w_matrix(my_class_weights)          
print(w_matrix)                              
#w_matrix=np.array([[1, 1, 1, 1, 1],
#                   [1000, 1, 1, 1, 1],
#                   [1000, 1, 1, 1, 1],
#                   [1000, 1, 1, 1, 1],
#                   [1000, 1, 1, 1, 1],
#        ])       
ncce = partial(w_categorical_crossentropy, weights=w_matrix)
auc_roc = as_keras_metric(tf.metrics.auc)
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
def create_baseline_dense():
    model = Sequential()
    #model.add(LSTM(52,input_shape=(50,32), return_sequences=True))
    model.add(Dense(10, activation='sigmoid'))
    model.add(Flatten())
    model.add(Dense(10, activation='sigmoid'))    
    model.add(Dense(5, activation='softmax'))
    model.compile(loss=ncce, optimizer=sgd, metrics=['acc',auc_roc])
    return model

dense_m=create_baseline_dense()
print("Запуск модели")
dense_m.fit(data_4_X, data_4_y_cat,validation_data=(data_4_X_t, data_4_y_t_cat), epochs=10, batch_size=32)

from sklearn.metrics import confusion_matrix
y_v=dense_m.predict_classes(data_4_X_t)
print(confusion_matrix(data_4_y_t, y_v))