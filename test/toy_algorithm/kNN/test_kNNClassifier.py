'''
Created on 2018年1月24日

@author: chzone
'''

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from toy_algorithm.kNN.kNNClassifier import kNNClassifier

def test_kNNClassifier():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
    
    cls_sk = KNeighborsClassifier(n_neighbors=3)
    cls_sk.fit(X_train,y_train)
    y_predict_sk = cls_sk.predict(X_test)
    
    cls_me = kNNClassifier(k=3)
    cls_me.fit(X_train,y_train)
    y_predict_me = cls_me.predict(X_test)
    
    assert y_predict_me.shape == y_predict_sk.shape
    assert sum(y_predict_me == y_predict_sk) == len(y_predict_me)
    
