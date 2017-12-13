from random import choice
from math import sqrt
from copy import copy



def euclidianDistance(a,b):
    return sqrt(sum([(a[i]-b[i]) ** 2 for i, _ in enumerate(a)]))

class Knn:
    def fit(self,X_train,Y_train,k=3):
        self.category = list(set(Y_train))
        self.X = X_train
        self.Y = Y_train
        self.k = k
    
    def predict(self,X_test):
        l = list()
        for row in X_test:
            l.append(sorted([(label,euclidianDistance(row,point)) for label,point in zip(self.Y,self.X)],key=lambda tp: tp[1]))
        return self.closeKN(l)
    
    def closeKN(self,l):
        predictions = list()
        for i in l:
            i.sort(key=lambda tupl: tupl[1])
            closest = max([([x for x,_ in i[:self.k]].count(y),y) for y in self.category])
            predictions.append(closest[1])
        return predictions