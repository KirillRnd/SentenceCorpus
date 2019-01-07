# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 10:52:08 2019

@author: Кирилл
"""
import Levenshtein as lv

import numpy as np
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
        #inv_d = 1.0 / d
        gauss=r(0,s,size=d)
        #length = np.linalg.norm(gauss)
        #if length == 0.0:
        #    v = gauss
        #else:
        #    r = np.random.rand() ** inv_d
        #    v = np.multiply(gauss, r / length)
        return gauss
        #return v
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