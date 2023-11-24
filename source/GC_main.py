# 忽略警告信息
import warnings

import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn import preprocessing
from sklearn.model_selection import KFold

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

# 在当前目录下运行
import os
import sys

os.chdir(sys.path[0])

if __name__ == '__main__':
    dataset = "lungcancer"  # dataset = lymphoma,DriveFace,SMK_CAN_187,COIL,HAPT,lungcancer,prostate
    data = sio.loadmat("E:/pythonproject/fuzzy_feature_selection/data/" + dataset + ".mat")
    data2 = data[dataset]  # COIL用data,HAPT_no_Bayes用HAPT
    data = pd.DataFrame(data2)


    # data = pd.read_csv(r"E:\pythonproject\fuzzy_feature_selection\data\mfeat-factors.csv")

    X = np.array(data.iloc[:, :-1])
    Y = np.array(data.iloc[:, [-1]])
    RMSE_all = []
    P = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    #归一化
    # min_max_scaler = preprocessing.MinMaxScaler()
    # X = min_max_scaler.fit_transform(X)
    # Y = np.array(Y)


    for p in P:
        print("\n***************  p:\t", str(p) + '\t*****************')
        times = 0
        RMSE_mean = []
        New_sam = KFold(n_splits=5, shuffle=True, random_state=12)

        for train_index, test_index in New_sam.split(X):
            times += 1
            X_this_time = X[train_index]
            Y_this_time = Y[train_index]
            X_arr = np.array(X_this_time)

            print("*******************  times:\t", str(p), '   ', str(times) + '\t***************************')
            X_masked = mask_types_p(X_arr, p, 1)  # 随机缺失数据,p为数据缺失率
            X_masked_copy = np.copy(X_masked)
            X_fill0 = np.copy(X_masked)
            X_fillmean = np.copy(X_masked)
            X_this_time_y = np.hstack((X_this_time, Y_this_time))

            # 构建缓冲矩阵
            col = X_arr.shape[1]
            Buffer_Matrix_width = 30
            buf_start = 0
            buf_end = Buffer_Matrix_width
            X_new = np.empty((X_arr.shape[0], 0))

            while buf_start < col:
                print(" buf_start:\t", str(buf_start) )
                if buf_end >= col:
                    This_X = X_arr[:, buf_start:]
                    This_X_masked = X_masked[:, buf_start:]
                else:
                    This_X = X_arr[:, buf_start:buf_end]
                    This_X_masked = X_masked[:, buf_start:buf_end]

                all_cont_indices = get_cont_indices(This_X_masked)
                all_ord_indices = ~all_cont_indices  # 按位取反

                # this_row = This_X_masked.shape[0]  # 读取矩阵第一维长度？
                # this_col = This_X_masked.shape[1]  # 读取矩阵第二维长度？

                # setting hyperparameter
                # BATCH_SIZE = 256
                # batch_size_denominator = math.ceil(this_row / BATCH_SIZE)
                # max_iter = batch_size_denominator * 3
                # WINDOW_SIZE = math.ceil(this_row / 2)

                em = ExpectationMaximization()
                X_imp, sigma_rearranged = em.impute_missing(This_X_masked, all_cont_indices, all_ord_indices)

                isnan = np.isnan(This_X_masked)
                X_imp[~isnan] = 0
                This_X_masked[isnan] = 0
                This_X_new = X_imp + This_X_masked

                X_new = np.hstack((X_new, This_X_new))
                buf_start += Buffer_Matrix_width
                buf_end += Buffer_Matrix_width
            #计算RMSE
            num = np.sum(np.isnan(X_masked_copy))
            sum = np.sum(np.power(X_new - X_arr, 2))
            RMSE = np.sqrt(sum / num)
            RMSE_mean.append(RMSE)
            print("Final RMSE: \n", RMSE)


            X_new_train = np.hstack((X_new, Y_this_time))
            X_new_test = np.hstack((X[test_index], Y[test_index]))
            X_fill0 = np.where(np.isnan(X_fill0), 0, X_fill0)
            X_fill0 = np.hstack((X_fill0, Y_this_time))
            X_fillmean = pd.DataFrame(X_fillmean)
            X_fillmean = np.array(X_fillmean.fillna(X_fillmean.mean()))
            X_fillmean = np.hstack((X_fillmean, Y_this_time))
            #
            # #保存文件
            path_LF = '../data/' + dataset + '/' + str(p)
            if os.path.exists(path_LF) == False:
                os.makedirs(path_LF)

            savepath_train_ori = path_LF + '/' + '_' + 'train_ori_' + str(times) + '.csv'
            # 保存预测文件为CSV
            with open(savepath_train_ori, 'w', newline='') as f:
                writer = csv.writer(f)  # 构造写入器
                for i in range(X_this_time_y.shape[0]):
                    writer.writerow(X_this_time_y[i, :])

            savepath_train_lf = path_LF + '/' + '_' + 'train_GC_' + str(times) + '.csv'
            # 保存预测文件为CSV
            with open(savepath_train_lf, 'w', newline='') as f:
                writer = csv.writer(f)  # 构造写入器
                for i in range(X_new_train.shape[0]):
                    writer.writerow(X_new_train[i, :])

            savepath_train_0 = path_LF + '/' + '_' + 'train_0_' + str(times) + '.csv'
            # 保存预测文件为CSV
            with open(savepath_train_0, 'w', newline='') as f:
                writer = csv.writer(f)  # 构造写入器
                for i in range(X_fill0.shape[0]):
                    writer.writerow(X_fill0[i, :])

            savepath_train_mean = path_LF + '/' + '_' + 'train_mean_' + str(times) + '.csv'
            # 保存预测文件为CSV
            with open(savepath_train_mean, 'w', newline='') as f:
                writer = csv.writer(f)  # 构造写入器
                for i in range(X_fillmean.shape[0]):
                    writer.writerow(X_fillmean[i, :])

            savepath_test = path_LF + '/' + '_test_' + str(times) + '.csv'
            # 保存测试文件为CSV
            with open(savepath_test, 'w', newline='') as f:
                writer = csv.writer(f)  # 构造写入器
                for i in range(X_new_test.shape[0]):
                    writer.writerow(X_new_test[i, :])
            t = 1
        RMSE_all.append(np.mean(RMSE_mean))






