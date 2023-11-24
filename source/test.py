import numpy as np
import pandas as pd
from source.evaluation.analysistool import evaluationSVM,evaluationKNN,evaluationRF


# data = pd.read_csv(r"E:\pythonproject\fuzzy_feature_selection\data\result_fivefold_\leukemia\0.1\_train_ori_1.csv",header=None)


dataset = "madelon"

# f = open(r"E:\pythonproject\Gaussian_feature_selection\data\result\Yale.txt")             # 返回一个文件对象

f = open(r"E:\result\GC\madelon.txt")
model = 'GC'

result = [[],[],[],[],[]]
i1 = 0
i2 = 0
dataset_times = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
while True:
    line = f.readline()
    line = line.strip()
    if len(line) == 0:
        break
    line_int = eval(line)
    if type(line_int) is int:
        list_line = [line_int]
    else:
        list_line = list(line_int)

    result[i1] = list_line
    i1 = i1 + 1
    # print(i1, list_line)

    if i1 == 5:
        i1 = 0


        dataset_time = dataset_times[i2]
        i2 += 1
        path_train = '../data/' +dataset +'/' + str(dataset_time) + '/_train_'+ model+'_'
        path_test = '../data/' + dataset + '/0.1/_test_'
        # path_train = 'data/result_fivefold_/leukemia/' + str(dataset_time) + '/_train_' + model + '_'
        # path_test = 'data/result_fivefold_/leukemia/0.1/_test_'


        scoreKNN = []
        scoreSVM = []
        scoreRF = []
        number_feature = []
        for i in range(5):
            result_this = result[i]
            result_this = np.array(result_this) - 1
            times = str(i + 1)
            path_thistime_train = path_train + times + '.csv'
            path_thistime_test = path_test + times + '.csv'
            # data_test = pd.read_csv(path_thistime_test, header=None)

            data_train = pd.read_csv(path_thistime_train, header=None)
            data_test = pd.read_csv(path_thistime_test, header=None)
            data_train_select = np.array(data_train.iloc[:, result_this])
            data_test_select = np.array(data_test.iloc[:, result_this])
            train_label = np.array(data_train.iloc[:, -1])
            test_label = np.array(data_test.iloc[:, -1])
            # t1 = crossValidationKNN(data_test_select,test_label)
            # t2 = crossValidationSVM(data_test_select,test_label)
            # t3 = crossValidationRF(data_test_select,test_label)
            t1 = evaluationKNN(data_train_select, train_label, data_test_select, test_label)
            t2 = evaluationSVM(data_train_select, train_label, data_test_select, test_label)
            t3 = evaluationRF(data_train_select, train_label, data_test_select, test_label)
            # print('')
            scoreKNN.append(t1)
            scoreSVM.append(t2)
            scoreRF.append(t3)
            number_feature.append(len(result_this))

        mean_KNN, std_KNN = np.mean(scoreKNN), np.std(scoreKNN)
        mean_SVM, std_SVM = np.mean(scoreSVM), np.std(scoreSVM)
        mean_RF, std_RF = np.mean(scoreRF), np.std(scoreRF)
        mean_all = np.mean([mean_RF, mean_SVM, mean_KNN])

        # print('number of select feature: %.2f'%np.mean(number_feature))
        # print('KNN means:  %.2f ± %.2f\n'%(mean_KNN,std_KNN))
        # print('SVM means:  %.2f ± %.2f\n'%(mean_SVM,std_SVM))
        # print('RF means:  %.2f ± %.2f\n'%(mean_RF,std_RF))
        # print("mean: %.2f\n"%mean_all)
        print(dataset_time)
        print('%.2f' % np.mean(number_feature))
        print('%.2f ± %.2f' % (mean_KNN, std_KNN))
        print('%.2f ± %.2f' % (mean_SVM, std_SVM))
        print('%.2f ± %.2f' % (mean_RF, std_RF))
        print("%.2f\n" % mean_all)



f.close()
