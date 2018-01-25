import numpy as np
from collections import Counter


def kNNFunc(k, X_train, Y_train, x):
    
    assert X_train.shape[0] == Y_train.shape[0] , \
    '''the number of the samples(X_train) must 
    equals to the number of the targets(Y_train)'''
    assert 0 < k <= X_train.shape[0], \
    'k invalid'
    assert X_train.shape[1] == x.shape[0]
    
    distances = [np.sqrt(sum((sample - x)**2)) for sample in X_train]
    sorted_index = np.argsort(distances)
    topk_y = [Y_train[i] for i in sorted_index[:k]]
    # votes 中保存的是最短距离对应的标记，而不是直接保存距离
    votes = Counter(topk_y)
    return votes.most_common(1)[0][0]
    

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
    x = np.array([8.093607318, 3.365731514])
    predict_y = kNNFunc(6, X_train, y_train, x)
    print(predict_y)