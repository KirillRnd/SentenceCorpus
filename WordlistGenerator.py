# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 10:46:56 2019

@author: Кирилл
"""
import pandas as pd
import numpy as np
import os
import re
import AdditionalF
from sklearn.cross_validation import train_test_split
import pickle
from WordlistMy import WordList
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

def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

s = np.random.seed(42)



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
save_obj(ww.dict,'wordlist')
save_obj(data_str,'data_str')

# In[98]:
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
sizeOfSet=5000
IDs_train=list(np.random.randint(len(X_train),size=sizeOfSet))
data_4_X=np.zeros((sizeOfSet,4,100))
data_4_y=np.zeros((sizeOfSet))
y_train=list(y_train)
X_train=list(X_train)
F=createVecFromStringDefault(ww,size_t=100)
i=0
for Id in IDs_train:
    label=y_train[Id]
    para=X_train[Id]
    data_4_X[i]=F(para).T
    data_4_y[i]=classes.get(label)
    print(i)
    i=i+1
    
data_4_y_cat=keras.utils.to_categorical(data_4_y, num_classes=5)
sizeOfSet_test=2000
IDs_test=list(np.random.randint(len(X_test),size=sizeOfSet_test))
data_4_X_t=np.zeros((sizeOfSet_test,4,100))
data_4_y_t=np.zeros((sizeOfSet_test))
y_test=list(y_test)
X_test=list(X_test)
F=createVecFromStringDefault(ww,size_t=100)
i=0
for Id in IDs_test:
    label=y_test[Id]
    para=X_test[Id]
    data_4_X_t[i]=F(para).T
    data_4_y_t[i]=classes.get(label)
    print(i)
    i=i+1
data_4_y_t_cat=keras.utils.to_categorical(data_4_y_t, num_classes=5) 
 
save_obj(data_4_y_t,'data_4_y_t')
save_obj(data_4_y_t_cat,'data_4_y_t_cat')
save_obj(data_4_y,'data_4_y')
save_obj(data_4_y_cat,'data_4_y_cat')
save_obj(data_4_X_t,'data_4_X_t')
save_obj(data_4_X,'data_4_X')