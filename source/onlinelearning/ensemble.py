import numpy as np
from onlinelearning.ftrl_adp import *
from evaluation.toolbox import delabel_func,semi_kMeans
from evaluation.helpers import get_mae

def ensemble_imp(n,X_input,Z_input,Y_label,miss_label,Y_label_imp,decay_choice,contribute_error_rate,k,p_miss):     #集合后做出预测
    errors = []
    classifier_X = FTRL_ADP(decay=1.0, L1=0., L2=0., LP=1., adaptive=True, n_inputs=X_input.shape[1])
    classifier_Z = FTRL_ADP(decay=1.0, L1=0., L2=0., LP=1., adaptive=True, n_inputs=Z_input.shape[1])

    x_loss = 0
    z_loss = 0
    lamda = 0.5
    eta = 0.001

    #随机缺失标签
    #Y_label_miss = np.array(delabel_func(Y_label))
    sum_random = 0

    for row in range(n):
        # indices_ran = [i for i in range(X_input.shape[1])]
        # indices_imp = [i for i in range(X_input.shape[1])]
        indices = [i for i in range(X_input.shape[1])]
        x = X_input[row]
        y = Y_label_imp[row]

        # if k:
        #     y = Y_label_miss[row]       #接收到达的可能有缺失的标签
        #     if np.isnan(y):
        #         L = np.column_stack((X_input[row-10:row],Y_label_miss[row-10:row]))  #np.hstack((X_input[row-5:row],Y_label_miss[row-5:row]))
        #         Y_label_miss[row] = semi_kMeans(L,X_input[row])
        #         y = Y_label_miss[row]

        if (k and (row > 10)):
            n_random = np.random.rand()
            if (n_random < p_miss and k):
                miss_label[row] = True
                #Y_label_ran[row] = np.random.choice(a=[0,1])
                # y_ran = Y_label_ran[row]
                L = np.column_stack((X_input[row - 10:row], Y_label_imp[row - 10:row]))
                Y_label_imp[row] = semi_kMeans(L,X_input[row])
                y = Y_label_imp[row]

        # x_loss_imp, z_loss_imp, lamda_imp = predict_ran(classifier_X, classifier_Z, Z_input, x_loss_imp, z_loss_imp,
        #                                                 row, lamda_imp, eta, errors_imp, indices, x, y_imp,decay_choice, contribute_error_rate)
        #
        # x_loss_ran,z_loss_ran,lamda_ran = predict_ran(classifier_X, classifier_Z, Z_input, x_loss_ran, z_loss_ran,
        #                                               row, lamda_ran, eta, errors_ran, indices, x, y_ran,decay_choice, contribute_error_rate)

        p_x, decay_x,loss_x, w_x = classifier_X.fit(indices, x, y ,decay_choice,contribute_error_rate)   #做出预测

        z = Z_input[row]
        p_z, decay_z, loss_z, w_z = classifier_Z.fit(indices, z, y, decay_choice, contribute_error_rate)

        #公式10，集成后的预测
        p = sigmoid(lamda*np.dot(w_x,x)+(1.0-lamda)*np.dot(w_z,z))

        #更新参数lamda 公式11
        x_loss += loss_x
        z_loss += loss_z
        lamda = np.exp(-eta*x_loss)/(np.exp(-eta*x_loss)+np.exp(-eta*z_loss))

        error = [int(np.abs(y - p) > 0.5)]
        errors.append(error)

    if k:
        print("impution:",get_mae(Y_label,Y_label_imp))
    #np.savetxt("../dataset/Datalabel/Y_label_imp.txt",list(map(int,Y_label_imp)))

    ensemble_error = np.cumsum(errors) / (np.arange(len(errors)) + 1.0)     #cumsum 累加和

    return ensemble_error


def ensemble_ran(n, X_input, Z_input, Y_label, Y_label_ran,decay_choice, contribute_error_rate):  # 集合后做出预测
    errors = []
    classifier_X = FTRL_ADP(decay=1.0, L1=0., L2=0., LP=1., adaptive=True, n_inputs=X_input.shape[1])
    classifier_Z = FTRL_ADP(decay=1.0, L1=0., L2=0., LP=1., adaptive=True, n_inputs=Z_input.shape[1])

    x_loss = 0
    z_loss = 0
    lamda = 0.5
    eta = 0.001

    for row in range(n):
        indices = [i for i in range(X_input.shape[1])]
        x = X_input[row]
        y = Y_label_ran[row]

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

    print("random:", get_mae(Y_label, Y_label_ran))

    ensemble_error = np.cumsum(errors) / (np.arange(len(errors)) + 1.0)  # cumsum 累加和

    return ensemble_error

def ensemble(n, X_input, Z_input, Y_label,decay_choice, contribute_error_rate):  # 集合后做出预测
    errors = []
    classifier_X = FTRL_ADP(decay=1.0, L1=0., L2=0., LP=1., adaptive=True, n_inputs=X_input.shape[1])
    classifier_Z = FTRL_ADP(decay=1.0, L1=0., L2=0., LP=1., adaptive=True, n_inputs=Z_input.shape[1])

    x_loss = 0
    z_loss = 0
    lamda = 0.5
    eta = 0.001

    for row in range(n):
        indices = [i for i in range(X_input.shape[1])]
        x = X_input[row]
        y = Y_label[row]

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


    ensemble_error = np.cumsum(errors) / (np.arange(len(errors)) + 1.0)  # cumsum 累加和

    return ensemble_error

def ensemble_not_updata(n, X_input, Z_input, Y_label,miss_label,decay_choice, contribute_error_rate):  # 集合后做出预测
    errors = []
    classifier_X = FTRL_ADP(decay=1.0, L1=0., L2=0., LP=1., adaptive=True, n_inputs=X_input.shape[1])
    classifier_Z = FTRL_ADP(decay=1.0, L1=0., L2=0., LP=1., adaptive=True, n_inputs=Z_input.shape[1])

    x_loss = 0
    z_loss = 0
    lamda = 0.5
    eta = 0.001

    for row in range(n):
        indices = [i for i in range(X_input.shape[1])]
        x = X_input[row]
        y = Y_label[row]

        if not miss_label[row]:
            p_x, decay_x, loss_x, w_x = classifier_X.fit(indices, x, y, decay_choice, contribute_error_rate)  # 做出预测

            z = Z_input[row]
            p_z, decay_z, loss_z, w_z = classifier_Z.fit(indices, z, y, decay_choice, contribute_error_rate)

        # 公式10，集成后的预测
        p = sigmoid(lamda * np.dot(w_x, x) + (1.0 - lamda) * np.dot(w_z, z))

        # 更新参数lamda 公式11
        if not miss_label[row]:
            x_loss += loss_x
            z_loss += loss_z
            lamda = np.exp(-eta * x_loss) / (np.exp(-eta * x_loss) + np.exp(-eta * z_loss))

        error = [int(np.abs(y - p) > 0.5)]
        errors.append(error)


    ensemble_error = np.cumsum(errors) / (np.arange(len(errors)) + 1.0)  # cumsum 累加和

    return ensemble_error

def ensemble_adp(n, X_input, Z_input, Y_label,miss_label,decay_choice, contribute_error_rate):  # 集合后做出预测
    errors = []
    classifier_X = FTRL_ADP(decay=1.0, L1=0., L2=0., LP=1., adaptive=True, n_inputs=X_input.shape[1])
    classifier_Z = FTRL_ADP(decay=1.0, L1=0., L2=0., LP=1., adaptive=True, n_inputs=Z_input.shape[1])

    x_loss = 0
    z_loss = 0
    lamda = 0.5
    eta = 0.001

    for row in range(n):
        indices = [i for i in range(X_input.shape[1])]
        x = X_input[row]
        y = Y_label[row]


        p_x, decay_x, loss_x, w_x = classifier_X.fit_adp(indices, x, y, decay_choice, contribute_error_rate,miss_label[row])  # 做出预测

        z = Z_input[row]
        p_z, decay_z, loss_z, w_z = classifier_Z.fit_adp(indices, z, y, decay_choice, contribute_error_rate,miss_label[row])

        # 公式10，集成后的预测
        p = sigmoid(lamda * np.dot(w_x, x) + (1.0 - lamda) * np.dot(w_z, z))

        # 更新参数lamda 公式11
        x_loss += loss_x
        z_loss += loss_z
        lamda = np.exp(-eta * x_loss) / (np.exp(-eta * x_loss) + np.exp(-eta * z_loss))

        error = [int(np.abs(y - p) > 0.5)]
        errors.append(error)


    ensemble_error = np.cumsum(errors) / (np.arange(len(errors)) + 1.0)  # cumsum 累加和

    return ensemble_error


def logistic_loss(p,y):
    return (1/np.log(2.0))*(-y*np.log(p)-(1-y)*np.log(1-p))

def sigmoid(x):
    if x >= 0:
        return 1.0 / (1 + np.exp(-x))
    else:
        return np.exp(x) / (1 + np.exp(x))