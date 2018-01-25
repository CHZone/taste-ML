'''
Created on 2018年1月23日

@author: chzone
'''
import numpy as np
from collections import Counter
from bokeh import __main__


class kNNClassifier:

    def __init__(self, k):
        assert 0 < k , 'invalid k'
        self.k = k  # 漏掉self
        self._X_train = None
        self._Y_train = None
        
    def fit(self, X_train, Y_train):
        assert self.k <= X_train.shape[0], 'the number of samples is to small'
        assert X_train.shape[0] == Y_train.shape[0], \
        '''the number of samples(X_train) must 
        equals to the number of target(Y_train)'''
        self._X_train = X_train
        self._Y_train = Y_train
        return self
        
    def predict(self, X):
        '''
        make sure the X.shape[1] equals to X_train.shape[1]
        '''
        assert self._X_train is not None and self._Y_train is not None, \
        'must fit before predict'
        assert X.shape[1] == self._X_train.shape[1], \
        'the number of features must  be equal'
        
        # 要返回np.array
        predicts = [self._predict(x) for x in X]
        return np.array(predicts)
    
    def _predict(self, x):
        assert x.shape[0] == self._X_train.shape[1], \
        '预测样本特征数与模型特征数不一致'
        distance = [np.sqrt(sum((sample - x) ** 2))for sample in self._X_train]
        sorted_index = np.argsort(distance)
        topk_y = sorted_index[:self.k]
        votes = Counter([self._Y_train[i] for i in topk_y])
        return votes.most_common(1)[0][0]
        
    def __repr__(self):
        return "KNN(k=%d)" % self.k
    
    
if __name__=='__main__':
    raw_data_X = [[3.393533211, 2.331273381],
              [3.110073483, 1.781539638],
              [1.343808831, 3.368360954],
              [3.582294042, 4.679179110],
              [2.280362439, 2.866990263],
              [7.423436942, 4.696522875],
              [5.745051997, 3.533989803],
              [9.172168622, 2.511101045],
              [7.792783481, 3.424088941],
              [7.939820817, 0.791637231]
             ]
    raw_data_y = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    
    X_train = np.array(raw_data_X)
    y_train = np.array(raw_data_y)
    
    clf = kNNClassifier(k=3)
    clf.fit(X_train, y_train)
    x = np.array([8.093607318, 3.365731514])
    predict_y = clf.predict(x.reshape(1, 2))
    print(predict_y)
        
