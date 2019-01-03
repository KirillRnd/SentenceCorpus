
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
        v=np.array([r(0,s),r(0,s),r(0,s),r(0,s)])
        return v
    def GetRndVec(self,words,create=False,ret=True):#словарь строится за n!/k! операций, где n - кол-во слов. Переделать
        def GetOne(word,create=False,ret=True):
            d=self.FindClosest(word)
            p=np.exp(-d.get('dist')/5)
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
    print(len(arr_tmp))
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
F=createVecFromStringDefault(ww,size_t=100)
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
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=TRAIN_SIZE, random_state=42)

print("Данные готовы")



# In[95]:
def NewF(string):
    return createVecFromString(string,ww,size_t=100)

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, X_func, batch_size=32, dim=(32,32,32), n_channels=1,
                 n_classes=5, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.X_func = X_func
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = self.X_func(X_train[ID]).reshape(self.dim)#применяем вышесозданную ф-ю

            # Store class
            y[i] = self.labels[ID]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)


# In[97]:


params = {'dim': (100,4,1),
          'batch_size': 64,
          'n_classes': 5,
          'n_channels': 1,
          'shuffle': True,
          'X_func': NewF
         }


IDs_train=np.random.randint(len(X_train),size=1000)
IDs_test=np.random.randint(len(X_test),size=1000)



training_generator = DataGenerator(IDs_train, y_train, **params)
validation_generator  = DataGenerator(IDs_test, y_test, **params)




model = Sequential()
model.add(Dense(40, input_shape=(100,4,1), activation='sigmoid'))
model.add(Dense(40, activation='sigmoid'))
model.add(Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

print("Запуск модели")

model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=False,
                    workers=1)

model.save('mod.hdf5')