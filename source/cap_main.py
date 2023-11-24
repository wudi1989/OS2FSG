
#忽略警告信息
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
import sys
sys.path.append("..")
from em.online_expectation_maximization import OnlineExpectationMaximization
from evaluation.helpers import *
from onlinelearning.online_learning import *
from onlinelearning.ensemble import *
import math
from evaluation.toolbox import *
import csv
from source.em.expectation_maximization import ExpectationMaximization



#在当前目录下运行
import os
import sys
os.chdir(sys.path[0])

if __name__ == '__main__':
    dataset = "credit" # australian,ionosphere,german,diabetes,wbc,wdbc,credit

    #getting  hyperparameter
    contribute_error_rate, window_size_denominator, batch_size_denominator, decay_coef_change,decay_choice,shuffle =get_cap_hyperparameter(dataset)
    MASK_NUM = 1
    X = pd.read_csv("../dataset/MaskData/"+dataset+"/X_process.txt",sep=" " ,header=None)
    Y_label = pd.read_csv("../dataset/Datalabel/" + dataset + "/Y_label.txt", sep=' ', header=None)


    X_masked = mask_types(X,MASK_NUM,seed=1) #arbitrary setting Nan
    # X_masked = np.array(pd.read_csv(r'C:\Users\LENOVO\Desktop\abs_angle.csv',header=None))

    X = X.values
    Y_label = Y_label.values


    all_cont_indices = get_cont_indices(X_masked)
    all_ord_indices = ~all_cont_indices   #按位取反

    n = X_masked.shape[0]           #读取矩阵第一维长度？
    feat = X_masked.shape[1]        #读取矩阵第二维长度？
    Y_label = Y_label.flatten()     #将矩阵降至1维？

    #setting hyperparameter
    BATCH_SIZE = 256
    batch_size_denominator = math.ceil(n/BATCH_SIZE)
    max_iter = batch_size_denominator * 2
    WINDOW_SIZE = math.ceil(n/2)

    # max_iter = batch_size_denominator * 2
    # BATCH_SIZE = math.ceil(n / batch_size_denominator)      #返回>=的最小整数
    # WINDOW_SIZE = math.ceil(n / window_size_denominator)
    # this_decay_coef = 0.5




    # start online copula imputation
    oem = OnlineExpectationMaximization(all_cont_indices, all_ord_indices, window_size=WINDOW_SIZE)
    j = 0
    X_imp = np.empty(X_masked.shape)        #随机生成数组
    Z_imp = np.empty(X_masked.shape)
    X_masked = np.array(X_masked)
    while j <= max_iter:
        start = (j * BATCH_SIZE) % n
        end = ((j + 1) * BATCH_SIZE) % n
        if end < start:
            indices = np.concatenate((np.arange(end), np.arange(start, n, 1)))
        else:
            indices = np.arange(start, end, 1)
        Z_imp[indices, :], X_imp[indices, :] = oem.partial_fit_and_predict(X_masked[indices, :], max_workers=1)
        j += 1

    isnan = np.isnan(X_masked)
    X_masked_copy = np.copy(X_masked)
    X_imp[~isnan] = 0
    X_masked_copy[isnan] = 0
    X_new = X_imp + X_masked_copy

    se2 = np.square(X - X_new)
    sum = np.sum(se2[~np.isnan(se2)])
    number = np.sum(np.isnan(X_masked)) - np.sum(np.isnan(X))
    mse2 = sum / number
    rmse2 = np.sqrt(mse2)



    # 设置保存路径
    path_x = "E:/pythonproject/"
    # 判断文件是否存在，不存在，就创建
    if os.path.exists(path_x) == False:
        os.makedirs(path_x)

    savepath_x = path_x + '/abs_angle'  + '.csv'
    # 保存预测文件为CSV
    with open(savepath_x, 'w', newline='') as f:
        writer = csv.writer(f)  # 构造写入器
        for i in range(X_imp.shape[0]):
            writer.writerow(X_imp[i, :])


    em = ExpectationMaximization()
    X_imp1, sigma_rearranged = em.impute_missing(X_masked,all_cont_indices,all_ord_indices)

    se1 = np.square(X - X_imp1)
    sum = np.sum(se1[~np.isnan(se1)])
    number =  np.sum(np.isnan(X_masked))-np.sum(np.isnan(X))
    mse1 = sum/number
    rmse1 = np.sqrt(mse1)

    s = 0
    e=0
    abss = 0
    for i in range(X.shape[0]):
        for k in range(X.shape[1]):
            rui = X_masked[i, k]
            if np.isnan(rui) and ~np.isnan(X[i,k]):
                e = e + pow((X[i, k] - X_imp1[i, k]), 2)
                abss = abss + np.abs(X[i, k] - X_imp1[i, k])
                s += 1
    rmse = math.sqrt(e / s)

    a = 3

'''
    # get the error of FOBOS and SVM
    temp = np.ones((n, 1))
    X_masked = pd.DataFrame(X_masked)
    X_zeros = X_masked.fillna(value=0)
    X_input_zero = np.hstack((temp, X_zeros))   #水平堆叠
    if shuffle == True:
        perm = np.arange(n)
        np.random.seed(1)
        np.random.shuffle(perm)     #序列元素随机排序
        Y_label = Y_label[perm]
        X_input_zero = X_input_zero[perm]

    X_Zero_CER, svm_error = generate_X(n, X_input_zero, Y_label, decay_choice, contribute_error_rate)


    #get the error of latent space
    temp_zim = np.ones((n, 1))
    X_input_z_imp = np.hstack((temp, Z_imp))        #水平方向叠加建立新矩阵
    if shuffle == True:
        perm = np.arange(n)
        np.random.seed(1)       #生成相同的随机
        np.random.shuffle(perm)     #随机打乱所有元素值
        X_input_z_imp = X_input_z_imp[perm]

    Y_label_ran = Y_label.copy()
    Y_label_imp = Y_label.copy()
    miss_label = np.full(len(Y_label),False)

    #Z_imp_CER = generate_cap(n,X_input_z_imp,Y_label,decay_choice,contribute_error_rate)
    # Z_imp_CER_semi = ensemble(n,X_input_z_imp,X_input_zero,Y_label,decay_choice,contribute_error_rate,1,p)  #半监督


    Z_imp_CER_semi = ensemble_imp(n,X_input_z_imp,X_input_zero,Y_label,miss_label,Y_label_imp,decay_choice,contribute_error_rate,1,p)
    np.savetxt("../dataset/Datalabel/" + dataset + "/Y_label_imp.txt",Y_label_imp)

    Z_imp_CER_adp = ensemble_adp(n, X_input_z_imp, X_input_zero, Y_label, miss_label, decay_choice,
                                  contribute_error_rate)
    Z_imp_CER = ensemble(n, X_input_z_imp, X_input_zero, Y_label, decay_choice, contribute_error_rate)  # 全监督
    #miss_label = np.full(len(Y_label), False)
    #Z_imp_CER_not_updata = ensemble_not_updata(n,X_input_z_imp,X_input_zero,Y_label,miss_label,decay_choice,contribute_error_rate)

    #draw_cap_error_picture(Z_imp_CER,X_Zero_CER,svm_error,dataset)  #全监督下的绘图
    draw_cap_error_picture(Z_imp_CER_adp, Z_imp_CER_semi, Z_imp_CER,svm_error,dataset)  #对比半监督与全监督

'''