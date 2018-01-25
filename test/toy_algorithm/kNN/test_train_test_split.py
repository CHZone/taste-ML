'''
Created on 2018年1月24日

@author: chzone
'''

from sklearn import datasets
from sklearn.model_selection import train_test_split as sk_train_test_split
from toy_algorithm.kNN.train_test_split import train_test_split


def test_train_test_split():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.2, seed=666)
    X_train2, X_test2, y_train2, y_test2 = sk_train_test_split(X, y, test_size=0.2, random_state=666)
    assert X_test.shape == X_test2.shape
    assert y_test.shape == y_test2.shape
    assert X_train.shape == X_train2.shape
    assert y_train.shape == y_train2.shape
