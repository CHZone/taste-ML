'''
Created on 2018年1月23日

@author: chzone
'''
import numpy as np


def train_test_split(X, y, test_ratio=0.2, seed=None):
    assert X.shape[0] == y.shape[0], \
    '实例与标记数不一致'
    assert 0.0 < test_ratio < 1.0, 'test_ratio=%f invalid' % test_ratio
    
    if seed is not None:
        np.random.seed(seed)
    
    shuffled_indexes = np.random.permutation(len(X))
    test_size = int(len(X) * test_ratio)
    X_test = X[shuffled_indexes[:test_size]]
    y_test = y[shuffled_indexes[:test_size]]
    
    X_train = X[shuffled_indexes[test_size:]]
    y_train = y[shuffled_indexes[test_size:]]
    
    return  X_train, X_test, y_train, y_test

