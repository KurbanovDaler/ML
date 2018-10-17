import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from functools import partial
from random import random
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict
from collections import Counter

df = load_iris()
df.data.shape

def f_hash(w,r,b,x):
    return int((np.dot(w,x)+b)/r)

def euclidianDistance(x, y):
    res = 0
    #print(x)
    for i in range(x.shape[0]):
        res += pow(x[i] - y[0][i], 2)
    return res
        
class KNNHash(object):
    def __init__(self,m,L,nn):
        self.m = m
        self.L = L
        self.nn = nn

    def fit(self,X,y):
        self.t_hh = [] #hash table
        for j in range(self.L):
            f_hh = [] #compositional hash function
            for i in range(self.m):
                w = np.random.rand(1,X[0].shape[0]) #  weights of a hash function
                f_hh.append(partial(f_hash,w = w,r=random(),b=random())) # list of initialized hash function
            self.t_hh.append(
                (defaultdict(list),f_hh)
            )
        for n in range(X.shape[0]): 
            for j in range(self.L):
                ind = 0
                for i in range(self.m):
                    ind = ind + self.t_hh[j][1][i](x=X[n]) #calculation of index in hash table, simply sum of all hash func
                self.t_hh[j][0][ind].append((X[n],y[n])) #saving sample into corresponding index
    
    def predict(self,u):        
        for j in range(self.L):
            inds = []
            distances = []
            for i in range(self.m):
                inds.append(self.t_hh[j][1][i](x=u))
            cntr = Counter([outp for inpt,outp in self.t_hh[j][0][sum(inds)]])
            print(cntr)
            for similar in self.t_hh[j][0][sum(inds)]:
                distances.append((euclidianDistance(u, similar), similar[1]))
            sorted(distances)    
            print(distances[0])
            #Here you must put your code, extend the method with distance function and calculation with unknown sample "u"
            #Develop the rest part of kNN predict method that was discussed at the lecture
            
scaler = MinMaxScaler()
scaler.fit(df.data)
x = scaler.transform(df.data)
y = df.target

knnhash = KNNHash(4,4,4)
test1x = x[0]
test2x = x[75]
test3x = x[149]

test1y = y[0]
test2y = y[75]
test3y = y[149]
x = np.delete(x,[0,75,149],axis=0)
y = np.delete(y,[0,75,149],axis=0)

knnhash.fit(x,y)



print(test1y)
knnhash.predict(test1x)
print("-------------")
print(test2y)
knnhash.predict(test2x)
print("-------------")
print(test3y)
knnhash.predict(test3x)
