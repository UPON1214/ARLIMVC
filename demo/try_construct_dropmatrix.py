import numpy as np
from numpy.random import randint
from sklearn.preprocessing import OneHotEncoder
import scipy.io



def get_mask(data_len, missing_rate=0.5, view_num=3):
    """Randomly generate incomplete data information, simulate partial view data with complete view data"""
    full_matrix = np.ones((int(data_len * (1 - missing_rate)), view_num))
    alldata_len = data_len - int(data_len * (1 - missing_rate))
    if alldata_len != 0:
        one_rate = 1.0 - missing_rate
        if one_rate <= (1 / view_num):
            enc = OneHotEncoder()  # n_values=view_num
            view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray()
            full_matrix = np.concatenate([view_preserve, full_matrix], axis=0)
            choice = np.random.choice(full_matrix.shape[0], size=full_matrix.shape[0], replace=False)
            matrix = full_matrix[choice]
            return matrix
        error = 1
        if one_rate == 1:
            matrix = randint(1, 2, size=(alldata_len, view_num))
            full_matrix = np.concatenate([matrix, full_matrix], axis=0)
            choice = np.random.choice(full_matrix.shape[0], size=full_matrix.shape[0], replace=False)
            matrix = full_matrix[choice]
            return matrix
        while error >= 0.005:
            enc = OneHotEncoder()  # n_values=view_num
            view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray()
            one_num = view_num * alldata_len * one_rate - alldata_len
            ratio = one_num / (view_num * alldata_len)
            matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(np.int_)
            a = np.sum(((matrix_iter + view_preserve) > 1).astype(np.int_))
            one_num_iter = one_num / (1 - a / one_num)
            ratio = one_num_iter / (view_num * alldata_len)
            matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(np.int_)
            matrix = ((matrix_iter + view_preserve) > 0).astype(np.int_)
            ratio = np.sum(matrix) / (view_num * alldata_len)
            error = abs(one_rate - ratio)
        full_matrix = np.concatenate([matrix, full_matrix], axis=0)

    choice = np.random.choice(full_matrix.shape[0], size=full_matrix.shape[0], replace=False)
    matrix = full_matrix[choice]
    return matrix

'''data = scipy.io.loadmat('../data/' + 'COIL20' + '/' + 'COIL20' + '.mat')
X = data['X'].T.squeeze()
n_views = X.shape[0]

droppedmatrix = get_mask(X[0].shape[0], missing_rate=0.3, view_num=3)
print(droppedmatrix)
print(droppedmatrix.shape)
# 计算矩阵中0的总体个数
total_zeros = np.sum(droppedmatrix == 0)

# 计算每一列中0的个数
column_zeros = np.sum(droppedmatrix == 0, axis=0)

print(f"总的0的个数: {total_zeros}")
print(f"每一列中0的个数: {column_zeros}")'''