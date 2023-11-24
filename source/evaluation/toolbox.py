import numpy as np
import pandas as pd

# 随机去除标签
def delabel_func(Y_label):
    delabel = []
    p = 0.8
    np.random.seed(3)
    delabel.append(np.random.choice(a=[False, True], size=len(Y_label), p=[p, 1 - p]))
    delabel[0][0:10] = True             #前10个标签是全的
    Y_label_miss = list(map(float,Y_label))
    for i in range(len(Y_label)):
        if not delabel[0][i]:
            Y_label_miss[i] = np.nan
    print(np.isnan(Y_label_miss).sum())
    return Y_label_miss

def predict_ran(classifier_X,classifier_Z,Z_input,x_loss,z_loss,row,lamda,eta,errors,indices, x, y ,decay_choice,contribute_error_rate):

    p_x, decay_x, loss_x, w_x = classifier_X.fit(indices, x, y, decay_choice, contribute_error_rate)  # 做出预测
    p_x1, decay_x1, loss_x1, w_x1 = classifier_X.fit(indices, x, y, decay_choice, contribute_error_rate)

    z = Z_input[row]
    p_z, decay_z, loss_z, w_z = classifier_Z.fit(indices, z, y, decay_choice, contribute_error_rate)

    # 公式10，集成后的预测
    p = sigmoid(lamda * np.dot(w_x, x) + (1.0 - lamda) * np.dot(w_z, z))

    # 更新参数lamda 公式11
    x_loss += loss_x
    z_loss += loss_z
    lamda = np.exp(-eta * x_loss) / (np.exp(-eta * x_loss) + np.exp(-eta * z_loss))

    error = [int(np.abs(y - p) > 0.5)]
    errors.append(error)


    # if k:
    #     print(get_mae(Y_label, Y_label_imp))
    #     print(sum_random)
    return x_loss,z_loss,lamda

def predict_imp(classifier_X,classifier_Z,Z_input,x_loss,z_loss,row,lamda,eta,errors,indices, x, y ,decay_choice,contribute_error_rate):

    p_x, decay_x, loss_x, w_x = classifier_X.fit(indices, x, y, decay_choice, contribute_error_rate)  # 做出预测

    z = Z_input[row]
    p_z, decay_z, loss_z, w_z = classifier_Z.fit(indices, z, y, decay_choice, contribute_error_rate)

    # 公式10，集成后的预测
    p = sigmoid(lamda * np.dot(w_x, x) + (1.0 - lamda) * np.dot(w_z, z))

    # 更新参数lamda 公式11
    x_loss += loss_x
    z_loss += loss_z
    lamda = np.exp(-eta * x_loss) / (np.exp(-eta * x_loss) + np.exp(-eta * z_loss))

    error = [int(np.abs(y - p) > 0.5)]
    errors.append(error)


    # if k:
    #     print(get_mae(Y_label, Y_label_imp))
    #     print(sum_random)
    return x_loss,z_loss,lamda

def distEclud(vecA, vecB):
    '''
    输入：向量A和B
    输出：A和B间的欧式距离
    '''
    return np.sqrt(sum(np.power(vecA - vecB, 2)))


def newCent(L):
    '''
    输入：有标签数据集L
    输出：根据L确定初始聚类中心
    '''
    centroids = []
    label_list = np.unique(L[:, -1])
    for i in label_list:
        L_i = L[(L[:, -1]) == i]
        cent_i = np.mean(L_i, 0)
        centroids.append(cent_i[:-1])
    return np.array(centroids)

#半监督补全标签
def semi_kMeans(L, U, distMeas=distEclud, initial_centriod=newCent):
    '''
    输入：有标签数据集L（最后一列为类别标签）、无标签数据集U（无类别标签）
    输出：聚类结果
    '''
    dataSet = np.vstack((L[:, :-1], U))  #合并L和U
    label_list = np.unique(L[:, -1])        #聚类中心的类别
    k = len(label_list)  # L中类别个数
    m = np.shape(dataSet)[0]

    clusterAssment = np.zeros(m)  # 初始化样本的分配
    centroids = initial_centriod(L)  # 确定初始聚类中心
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):  # 将每个样本分配给最近的聚类中心
            minDist = np.inf;
            minIndex = -1
            for j in range(k):
                distJI = distMeas(centroids[j, :], dataSet[i, :])
                if distJI < minDist:
                    minDist = distJI;
                    minIndex = j
            if clusterAssment[i] != minIndex: clusterChanged = True
            clusterAssment[i] = minIndex
    return clusterAssment[-1]


def DictCompare(W, X, op = 'intersection'):
    if op == 'intersection':
        shared_W_keys = list(W.keys() & X.keys())
        shared_W_keys.sort()
        W = DictToList(W, shared_W_keys)
        X = DictToList(X, shared_W_keys, WorX='X')
        return shared_W_keys, W, X # keys and corresponding values
    if op == 'extraW':
        extra_W_keys = np.sort(np.array(list(W.keys() - X.keys())))
        extra_W_values = DictToList(W, extra_W_keys)
        return extra_W_keys, extra_W_values

def DictToList(dict, indexList = None, WorX = 'weight'):
    df = pd.DataFrame(dict)
    if indexList is not None:
        df_sel = df[indexList]
        if WorX != 'weight':
            return df_sel.values.reshape(1, -1)
        else:
            return df_sel.values.reshape(-1, 1)
    else:
        if WorX != 'weight':
            return df.values.reshape(1, -1)
        else:
            return df.values.reshape(-1, 1)

def MatrixInDict(matrix, dict):
    matrixTemp = matrix.copy()
    '''Always take the full feature space as the dimension of mapped vector'''
    for (r,row) in enumerate(matrix):
        for (c,col) in enumerate(row):
            if dict.get(r) is not None:
                key_new_feature = dict.get(r)
                if key_new_feature is not None:
                    key_new_to_all = key_new_feature.get(c)
                    matrixTemp[r, c] = key_new_to_all
    return matrixTemp

def sigmoid(x):
    if x >= 0:
        return 1.0 / (1 + np.exp(-x))
    else:
        return np.exp(x) / (1 + np.exp(x))