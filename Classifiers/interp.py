from numpy import interp
import numpy as np
import pandas as pd

# mean_recall = [1.        , 0.98989899, 0.97979798, 0.96969697, 0.95959596,
#        0.94949495, 0.93939394, 0.92929293, 0.91919192, 0.90909091,
#        0.8989899 , 0.88888889, 0.87878788, 0.86868687, 0.85858586,
#        0.84848485, 0.83838384, 0.82828283, 0.81818182, 0.80808081,
#        0.7979798 , 0.78787879, 0.77777778, 0.76767677, 0.75757576,
#        0.74747475, 0.73737374, 0.72727273, 0.71717172, 0.70707071,
#        0.6969697 , 0.68686869, 0.67676768, 0.66666667, 0.65656566,
#        0.64646465, 0.63636364, 0.62626263, 0.61616162, 0.60606061,
#        0.5959596 , 0.58585859, 0.57575758, 0.56565657, 0.55555556,
#        0.54545455, 0.53535354, 0.52525253, 0.51515152, 0.50505051,
#        0.49494949, 0.48484848, 0.47474747, 0.46464646, 0.45454545,
#        0.44444444, 0.43434343, 0.42424242, import pandas as pd
# import numpy as np
# import random
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import KFold
# from sklearn.metrics import accuracy_score, precision_score, f1_score
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import roc_curve, auc
# import matplotlib.pyplot as plt
# from sklearn.metrics import precision_recall_curve
#
# # import dataset
# input_path = r'E:\论文\DTI\LTP_V2\data\EN_P.csv'
# input_path1 = r'E:\论文\DTI\LTP_V2\data\EN_N.csv'
# input_data = pd.read_csv(input_path, header=None)
# input_data1 = pd.read_csv(input_path1, header=None)
# input0 = np.array(input_data)
# input1 = np.array(input_data1)
# input = np.vstack((input0, input1))  # 按照列合并矩阵
# index = [i for i in range(5850)]
# random.shuffle(index)
# input_new = input[index]
# X = input_new[:, 1:1394]
# Y = input_new[:, 0]
#
#
# from sklearn.ensemble import RandomForestClassifier
# rfc = RandomForestClassifier(n_estimators=100,  oob_score=True, random_state=42)
#
# KF = KFold(n_splits=5)  # 建立5折交叉验证方法  查一下KFold函数的参数
#
# X_train = []
# Y_train = []
# X_test = []
# Y_test = []
#
# Real_data = []
# Predict_data = []
# Real_data1 = []
# for train_index, test_index in KF.split(X):
#     print("TRAIN:", train_index, "TEST:", test_index)
#     x_train, x_test = X[train_index], X[test_index]
#     y_train, y_test = Y[train_index], Y[test_index]
#     X_train.append(x_train)
#     Y_train.append(y_train)
#     X_test.append(x_test)
#     Y_test.append(y_test)
#
#
#     gbdt_grid_results = rfc.fit(x_train, y_train)
#
#     y_pre = rfc.predict(x_test)
#     Real_data.append(y_pre)
#     Predict_data.append(y_test)
#
#     y_pre1 = rfc.predict_proba(x_test)
#     Real_data1.append(y_pre1)
#
#
#     f1 = f1_score(y_test, y_pre, average='micro')
#     print("the f1 score: %.2f" % f1)
#     C2 = confusion_matrix(y_test, y_pre, labels=[0, 1])
#     print(C2)
#     print(C2.ravel())
#     C = C2.ravel()
#     MCC = (C[0] * C[3] - C[1] * C[2]) / ((C[3] + C[1]) * (C[3] + C[2]) * (C[0] + C[1]) * (C[0] + C[2])) ** 0.5
#     Sen = C[3] / (C[3] + C[2])
#     Acc = (C[3] + C[0]) / (C[3] + C[0] + C[1] + C[2])
#     Pre = C[3] / (C[1] + C[3])
#     Spec = C[0] / (C[0] + C[1])
#     print("******************************************")
#     print("Acc:", Acc)
#     print("Pre:", Pre)
#     print("Sen:", Sen)
#     print("Spec:", Spec)
#     print("MCC:", MCC)
#
#     # print(classification_report(y, final_prediction, digits=4))  # Pre、Sen(recall)
# print("******************************************")
# fpr, tpr, threshold = roc_curve(Real_data[0], Predict_data[0])
# fpr1, tpr1, threshold1 = roc_curve(Real_data[1], Predict_data[1])
# fpr2, tpr2, threshold2 = roc_curve(Real_data[2], Predict_data[2])
# fpr3, tpr3, threshold3 = roc_curve(Real_data[3], Predict_data[3])
# fpr4, tpr4, threshold4 = roc_curve(Real_data[4], Predict_data[4])
#
# precision1, recall1, thresholds = precision_recall_curve(Predict_data[1], Real_data1[1][:, 1], pos_label=None, sample_weight=None)
#
#
# roc_auc = auc(fpr, tpr)
# print(roc_auc)
# roc_auc1 = auc(fpr1, tpr1)
# print(roc_auc1)
# roc_auc2 = auc(fpr2, tpr2)
# print(roc_auc2)
# roc_auc3 = auc(fpr3, tpr3)
# print(roc_auc3)
# roc_auc4 = auc(fpr4, tpr4)
# print(roc_auc4)  # 计算auc的值
# average_auc = (roc_auc + roc_auc1 + roc_auc2 + roc_auc3 + roc_auc4) / 5
# lw = 4
# fig = plt.figure(figsize=(10, 10))
# # plt.plot(fpr, tpr, color='darkorange', lw=lw, label='AUC_Fold1  (area = %0.3f)' % roc_auc)
# # plt.plot(fpr, tpr, color='black', lw=lw, label='1st fold = %0.4f' % roc_auc)
# # plt.plot(fpr1, tpr1, color='blue', lw=lw, label='2nd fold = %0.4f' % roc_auc1)
# # plt.plot(fpr2, tpr2, color='red', lw=lw, label='3rd fold = %0.4f' % roc_auc2)
# # plt.plot(fpr3, tpr3, color='yellow', lw=lw, label='4th fold = %0.4f' % roc_auc3)
# plt.plot(fpr4, tpr4, color='green', lw=lw, label='5th fold = %0.4f' % roc_auc4)  # 假正率为横坐标，真正率为纵坐标做曲线
# # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
#
# plt.plot(precision1, recall1, color='green', lw=lw, label='5th fold = %0.4f' % roc_auc4)
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.0])
# plt.text(0.22, 0.05, "Average AUC = %0.4f" % average_auc, size=34, alpha=1)
# plt.xlabel('1-Specificity', fontsize=30)
# plt.ylabel('Sensitivity', fontsize=30)
# # plt.title('Receiver operating characteristic example')
# plt.legend(loc="right", fontsize=26)
#
# ax = plt.gca()  # 获得坐标轴的句柄
# ax.spines['bottom'].set_linewidth(2)  ###设置底部坐标轴的粗细
# ax.spines['left'].set_linewidth(2)  ####设置左边坐标轴的粗细
# ax.spines['right'].set_linewidth(2)  ###设置右边坐标轴的粗细
# ax.spines['top'].set_linewidth(2)  ####设置上部坐标轴的粗细
#
# plt.tick_params(labelsize=24)
# plt.show()
#
# # np.savetxt(r"D:\QQ文件\Paper_2\conpare_AUC_data\LGBM\NR_Fpr.csv", fpr, delimiter=",")
# # np.savetxt(r"D:\QQ文件\Paper_2\conpare_AUC_data\LGBM\NR_Tpr.csv", tpr, delimiter=",")0.41414141, 0.4040404 ,
#        0.39393939, 0.38383838, 0.37373737, 0.36363636, 0.35353535,
#        0.34343434, 0.33333333, 0.32323232, 0.31313131, 0.3030303 ,
#        0.29292929, 0.28282828, 0.27272727, 0.26262626, 0.25252525,
#        0.24242424, 0.23232323, 0.22222222, 0.21212121, 0.2020202 ,
#        0.19191919, 0.18181818, 0.17171717, 0.16161616, 0.15151515,
#        0.14141414, 0.13131313, 0.12121212, 0.11111111, 0.1010101 ,
#        0.09090909, 0.08080808, 0.07070707, 0.06060606, 0.05050505,
#        0.04040404, 0.03030303, 0.02020202, 0.01010101, 0.        ]

mean_recall = np.linspace(0, 1, 100)

    # Linear interpolation method
# recall = [1., 0.98885512, 0.98074975, 0.97365755, 0.96352584,
#        0.95339412, 0.93819656, 0.92097264, 0.90982776, 0.89463019,
#        0.87943262, 0.86322188, 0.84194529, 0.82877406, 0.8064843,
#        0.78115502, 0.76190476, 0.7325228, 0.70618034, 0.66261398,
#        0.61702128, 0.57548126, 0.5035461, 0.41742655, 0.]

recall = np.array([1., 0.98885512, 0.98074975, 0.97365755, 0.96352584,
       0.95339412, 0.93819656, 0.92097264, 0.90982776, 0.89463019,
       0.87943262, 0.86322188, 0.84194529, 0.82877406, 0.8064843,
       0.78115502, 0.76190476, 0.7325228, 0.70618034, 0.66261398,
       0.61702128, 0.57548126, 0.5035461, 0.41742655, 0.])
recall1 = np.matrix.tolist(recall)

precision = [0.49672874, 0.56286044, 0.6057572 , 0.63769078, 0.66643308,
       0.6965211 , 0.7211838 , 0.74386252, 0.76360544, 0.78488889,
       0.8       , 0.815311  , 0.8335005 , 0.85297185, 0.86899563,
       0.88013699, 0.8973747 , 0.90943396, 0.91710526, 0.92897727,
       0.93981481, 0.95302013, 0.96317829, 0.97862233, 1.]
print(type(recall1))
print(type(precision))

recall1.reverse()
precision.reverse()

recall2 = np.array(recall1)
precision = np.array(precision)
print(type(recall2))
print(type(precision))

a = interp(mean_recall, recall2, precision, )
print(a)