# -*- coding: utf-8 -*-
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import time
import warnings
from sklearn.metrics import precision_recall_curve
from numpy import interp


mean_fpr = np.linspace(0, 1, 100)  # Linear interpolation method
mean_recall = np.linspace(1, 0, 100)  # Linear interpolation method

warnings.filterwarnings("ignore")

# Input complete data
input_path = r'E:\论文\分类器\Word2Vec_LNCRNA\feature_500\new_PN\P.xlsx'
input_path1 = r'E:\论文\分类器\Word2Vec_LNCRNA\feature_500\new_PN\N.xlsx'

input_data = pd.read_excel(input_path, header=None)
input_data1 = pd.read_excel(input_path1, header=None)

input0 = np.array(input_data)
input1 = np.array(input_data1)
input = np.vstack((input0, input1))  # Merge matrix by column

index = [i for i in range(9932)]
random.shuffle(index)
input_new = input[index]

# y = input[10:20,0]  # test
# y_new = input_new[10:20,0]

X = input_new[:, 1:301]
Y = input_new[:, 0]


print("********************************")
print("The shape of N_EN:", input0.shape)
print("The shape of P_EN:", input1.shape)
print("The shape of input:", input.shape)
print("The shape of input_new:", input_new.shape)
print("*type_X", type(X))
print("*X_shape:", X.shape)
print("*type_Y", type(Y))
print("*Y_shape:", Y.shape)
print("********************************")


# Get random subsets
def get_random_subset(iterable, k):
    subsets = []
    iteration = 0
    np.random.shuffle(iterable)
    subset = 0
    limit = len(iterable) / k
    while iteration < limit:
        if k <= len(iterable):
            subset = k
        else:
            subset = len(iterable)
        subsets.append(iterable[-subset:])
        del iterable[-subset:]
        iteration += 1
    return subsets


# Building a rotating forest model
def build_rotationtree_model(x_train, y_train, d, k):
    models = []  # Decision tree
    r_matrices = []  # Rotation matrix related to tree
    feature_subsets = []  # Feature subset used in iteration
    for i in range(d):
        # x, _, _, _ = train_test_split(x_train, y_train, test_size=0.25, random_state=7)
        x = x_train
        # Index of features
        feature_index = list(range(x.shape[1]))
        # Get the subsets of features
        random_k_subset = get_random_subset(feature_index, k)
        feature_subsets.append(random_k_subset)
        R_matrix = np.zeros((x.shape[1], x.shape[1]), dtype=float)  # Rotation matrix
        for each_subset in random_k_subset:
            pca = PCA()
            x_subset = x[:, each_subset]  # Extract the value of x corresponding to the index in the subset
            pca.fit(x_subset)  # Principal component analysis
            for ii in range(0, len(pca.components_)):
                for jj in range(0, len(pca.components_)):
                    R_matrix[each_subset[ii], each_subset[jj]] = pca.components_[ii, jj]
        x_transformed = x_train.dot(R_matrix)

        model = DecisionTreeClassifier()
        model.fit(x_transformed, y_train)

        models.append(model)
        r_matrices.append(R_matrix)
    return models, r_matrices, feature_subsets


def evaluation(real, pre):
    TP = 0
    FP = 0
    TN = 0
    FN = 0




# main

def model_worth(models, r_matrices, x, y):
    predicted_ys = []
    predicted_matrix = []
    final_score = []  # 存放所有得分
    P = []  # 存放每一个y
    Scores = []  # 存放概率
    for i, model in enumerate(models):
        x_mod = x.dot(r_matrices[i])
        score = model.predict_proba(x_mod)
        predicted_y = model.predict(x_mod)  # 直接预测为0/1

        # print("score.shape:", score[1, 0:2])
        # print("score.:", score[:, 1])
        predicted_ys.append(predicted_y)
        Scores.append(score)
        predicted_matrix = np.asmatrix(predicted_ys)  # 转化为矩阵 d*1755

    final_prediction = []
    a = []


    for i in range(len(models)):
        Scores[0] = Scores[0] + Scores[i]
    Scores[0] = Scores[0]/len(models)
    for ii in range(Scores[0].shape[0]):
        if Scores[0][ii, 0] > Scores[0][ii, 1]:
            final_prediction.append(0)
        else:
            final_prediction.append(1)
        a.append(Scores[0][ii, 1])

    C2 = confusion_matrix(y, final_prediction, labels=[0, 1])
    print(C2)
    print(C2.ravel())
    C = C2.ravel()
    MCC = (C[0]*C[3]-C[1]*C[2])/((C[3]+C[1])*(C[3]+C[2])*(C[0]+C[1])*(C[0]+C[2]))**0.5
    Sen = C[3]/(C[3]+C[2])
    Acc = (C[3]+C[0])/(C[3]+C[0]+C[1]+C[2])
    Pre = C[3]/(C[1]+C[3])
    Spec = C[0]/(C[0]+C[1])

    print("******************************************")
    print("Acc:", Acc)
    print("Pre:", Pre)
    print("Sen:", Sen)
    print("Spec:", Spec)
    print("MCC:", MCC)
    # print(classification_report(y, final_prediction, digits=4))  # Pre、Sen(recall)
    print("******************************************")
    return predicted_matrix, y, a


if __name__ == "__main__":
    x = X
    y = Y

    # Partition data set
    KF = KFold(n_splits=5)  # Establish 5-fold cross validation method
    Real_data = []
    Real_data1 = []
    Score = []

    for train_index, test_index in KF.split(x):
        print("TRAIN:", train_index, "TEST:", test_index)
        x_train, x_test_all = X[train_index], X[test_index]
        y_train, y_test_all = Y[train_index], Y[test_index]
        models, r_matrices, features = build_rotationtree_model(x_train, y_train, 22, 5)
        # Determine the number of trees K and feature subsets L
        # Utilize grid search method here
        predicted_matrix1, real, score = model_worth(models, r_matrices, x_test_all, y_test_all)

        Real_data.append(real)  # predicted results
        Real_data1.append(y_test_all)  # real results
        Score.append(score)
    print(Score[1][1])
    print(len(Real_data))
    tprs = []

    fpr, tpr, threshold = roc_curve(Real_data1[0], Score[0])
    tprs.append(interp(mean_fpr, fpr, tpr))
    fpr1, tpr1, threshold1 = roc_curve(Real_data1[1], Score[1])
    tprs.append(interp(mean_fpr, fpr1, tpr1))
    fpr2, tpr2, threshold2 = roc_curve(Real_data1[2], Score[2])
    tprs.append(interp(mean_fpr, fpr2, tpr2))
    fpr3, tpr3, threshold3 = roc_curve(Real_data1[3], Score[3])
    tprs.append(interp(mean_fpr, fpr3, tpr3))
    fpr4, tpr4, threshold4 = roc_curve(Real_data1[4], Score[4])  # 计算真正率和假正率
    tprs.append(interp(mean_fpr, fpr4, tpr4))
    # tprs.append(interp(mean_fpr, fpr4, tpr4))

    # Use interpolation method to generate average fpr and average fpr
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

    fig0 = plt.figure(figsize=(10, 10))
    # Plot the mean ROC curve
    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (area=%0.2f)' % mean_auc, lw=2, alpha=.8)
    plt.text(0.20, 0.05, "Average AUC = %0.4f" % mean_auc, size=34, alpha=1)
    plt.xlabel('1-Specificity', fontsize=30)
    plt.ylabel('Sensitivity', fontsize=30)
    ax = plt.gca()  # 获得坐标轴的句柄
    ax.spines['bottom'].set_linewidth(2)  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2)  ####设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2)  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)  ####设置上部坐标轴的粗细
    plt.show()
    # fig0.savefig(r'E:\论文\LncRNA-miRNA\final\mean_ROC.tif', dpi=600, format='tif')


    precisions = []
    # Prepare for PR curve
    precision1, recall1, thresholds = precision_recall_curve(Real_data1[0], Score[0], pos_label=None,
                                                             sample_weight=None)
    precision1 = np.matrix.tolist(precision1)  #逆序recall和precision
    recall1 = np.matrix.tolist(recall1)
    precision1.reverse()
    recall1.reverse()
    precisions.append(interp(mean_recall, recall1, precision1))

    precision2, recall2, thresholds2 = precision_recall_curve(Real_data1[1], Score[1], pos_label=None,
                                                             sample_weight=None)
    precision2 = np.matrix.tolist(precision2)
    recall2 = np.matrix.tolist(recall2)
    precision2.reverse()
    recall2.reverse()
    precisions.append(interp(mean_recall, recall2, precision2))

    precision3, recall3, thresholds3 = precision_recall_curve(Real_data1[2], Score[2], pos_label=None,
                                                             sample_weight=None)
    precision3 = np.matrix.tolist(precision3)  # 逆序recall和precision
    recall3 = np.matrix.tolist(recall3)
    precision3.reverse()
    recall3.reverse()
    precisions.append(interp(mean_recall, recall3, precision3))

    precision4, recall4, thresholds4 = precision_recall_curve(Real_data1[3], Score[3], pos_label=None,
                                                             sample_weight=None)
    precision4 = np.matrix.tolist(precision4)  # 逆序recall和precision
    recall4 = np.matrix.tolist(recall4)
    precision4.reverse()
    recall4.reverse()
    precisions.append(interp(mean_recall, recall4, precision4))

    precision5, recall5, thresholds5 = precision_recall_curve(Real_data1[4], Score[4], pos_label=None,
                                                             sample_weight=None)
    precision5 = np.matrix.tolist(precision5)  # 逆序recall和precision
    recall5 = np.matrix.tolist(recall5)
    precision5.reverse()
    recall5.reverse()
    precisions.append(interp(mean_recall, recall5, precision5))

    roc_auc = auc(fpr, tpr)
    print(roc_auc)
    roc_auc1 = auc(fpr1, tpr1)
    print(roc_auc1)
    roc_auc2 = auc(fpr2, tpr2)
    print(roc_auc2)
    roc_auc3 = auc(fpr3, tpr3)
    print(roc_auc3)
    roc_auc4 = auc(fpr4, tpr4)
    print(roc_auc4)  # Calculate the AUC value
    average_auc = (roc_auc+roc_auc1+roc_auc2+roc_auc3+roc_auc4)/5
    lw = 4

    # Plot ROC curve
    fig1 = plt.figure(figsize=(10, 10))
    # plt.plot(fpr, tpr, color='darkorange', lw=lw, label='AUC_Fold1  (area = %0.3f)' % roc_auc)
    plt.plot(fpr, tpr, color='black', lw=lw, label='1st fold  = %0.4f' % roc_auc)
    plt.plot(fpr1, tpr1, color='blue', lw=lw, label='2nd fold = %0.4f' % roc_auc1)
    plt.plot(fpr2, tpr2, color='red', lw=lw, label='3rd fold  = %0.4f' % roc_auc2)
    plt.plot(fpr3, tpr3, color='yellow', lw=lw, label='4th fold  = %0.4f' % roc_auc3)
    plt.plot(fpr4, tpr4, color='green', lw=lw, label='5th fold  = %0.4f' % roc_auc4)  # 假正率为横坐标，真正率为纵坐标做曲线

    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # print(fpr)
    # print(type(tpr))
    # print(fpr.shape)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.text(0.20, 0.05, "Average AUC = %0.4f" % average_auc, size=34, alpha=1)
    plt.xlabel('1-Specificity', fontsize=30)
    plt.ylabel('Sensitivity', fontsize=30)
    # plt.title('Receiver operating characteristic example')
    plt.legend(loc="right", bbox_to_anchor=(1, 0.43), fontsize=24)

    plt.tick_params(labelsize=24)  # Scale font size 13

    ###Set the thickness of the axis
    ax = plt.gca()  # 获得坐标轴的句柄
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    plt.show()
    # Save ROC curve
    # fig1.savefig(r'E:\论文\LncRNA-miRNA\final\ROC.tif', dpi=600, format='tif')



    # Plot PR curve
    fig2 = plt.figure(figsize=(10, 10))
    aupr = auc(recall1, precision1)
    aupr1 = auc(recall2, precision2)
    aupr2 = auc(recall3, precision3)
    aupr3 = auc(recall4, precision4)
    aupr4 = auc(recall5, precision5)

    plt.plot(recall1, precision1, color='black', lw=lw, label='1st fold  = %0.4f' % aupr)
    plt.plot(recall2, precision2, color='blue', lw=lw, label='2nd fold  = %0.4f' % aupr1)
    plt.plot(recall3, precision3, color='red', lw=lw, label='3rd fold  = %0.4f' % aupr2)
    plt.plot(recall4, precision4, color='yellow', lw=lw, label='4th fold  = %0.4f' % aupr3)
    plt.plot(recall5, precision5, color='green', lw=lw, label='5th fold  = %0.4f' % aupr4)

    average_aupr = (aupr+aupr1+aupr2+aupr3+aupr4)/5

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.text(0.20, 0.05, "Average AUPR = %0.4f" % average_aupr, size=34, alpha=1)
    plt.xlabel('Recall', fontsize=30)
    plt.ylabel('Precision', fontsize=30)
    # plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower left", bbox_to_anchor=(0, 0.23), fontsize=24)
    #, bbox_to_anchor=(0, 0.43)
    plt.tick_params(labelsize=24)

    ax = plt.gca()
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    plt.show()
    # Save PR curve
    # fig2.savefig(r'E:\论文\LncRNA-miRNA\final\PR.tif', dpi=600, format='tif')


    # Use interpolation method to generate average Recall and average precision
    mean_precision = np.mean(precisions, axis=0)
    mean_precision[0] = 1.0

    mean_aupr = auc(mean_recall, mean_precision)
    std_precision = np.std(precisions, axis=0)
    precisions_upper = np.minimum(mean_precision + std_precision, 1)
    precisions_lower = np.maximum(mean_precision - std_precision, 0)
    fig3 = plt.figure(figsize=(10, 10))
    # Plot the mean ROC curve
    plt.plot(mean_recall, mean_precision, color='b', label=r'Mean AUPR (area=%0.2f)' % average_aupr, lw=2, alpha=.8)
    plt.text(0.20, 0.05, "Average AUPR = %0.4f" % mean_aupr, size=34, alpha=1)
    plt.xlabel('Recall', fontsize=30)
    plt.ylabel('Precision', fontsize=30)
    ###设置坐标轴的粗细
    ax = plt.gca()  # 获得坐标轴的句柄
    ax.spines['bottom'].set_linewidth(2)  ###设置底部坐标轴的粗细
    ax.spines['left'].set_linewidth(2)  ###设置左边坐标轴的粗细
    ax.spines['right'].set_linewidth(2)  ###设置右边坐标轴的粗细
    ax.spines['top'].set_linewidth(2)  ###设置上部坐标轴的粗细
    plt.show()
    # fig3.savefig(r'E:\论文\LncRNA-miRNA\final\mean_PR.tif', dpi=600, format='tif')



    # save the average FPR and TPR

    # np.savetxt(r"E:\论文\LncRNA-miRNA\final\ROC\mean_Fpr.csv", mean_fpr, delimiter=",")
    # np.savetxt(r"E:\论文\LncRNA-miRNA\final\ROC\mean_Tpr.csv", mean_tpr, delimiter=",")
    # np.savetxt(r"E:\论文\LncRNA-miRNA\final\ROC\Fpr.csv", fpr, delimiter=",")
    # np.savetxt(r"E:\论文\LncRNA-miRNA\final\ROC\Tpr.csv", tpr, delimiter=",")
    # np.savetxt(r"E:\论文\LncRNA-miRNA\final\ROC\Fpr1.csv", fpr1, delimiter=",")
    # np.savetxt(r"E:\论文\LncRNA-miRNA\final\ROC\Tpr1.csv", tpr1, delimiter=",")
    # np.savetxt(r"E:\论文\LncRNA-miRNA\final\ROC\Fpr2.csv", fpr2, delimiter=",")
    # np.savetxt(r"E:\论文\LncRNA-miRNA\final\ROC\Tpr2.csv", tpr2, delimiter=",")
    # np.savetxt(r"E:\论文\LncRNA-miRNA\final\ROC\Fpr3.csv", fpr3, delimiter=",")
    # np.savetxt(r"E:\论文\LncRNA-miRNA\final\ROC\Tpr3.csv", tpr3, delimiter=",")
    # np.savetxt(r"E:\论文\LncRNA-miRNA\final\ROC\Fpr4.csv", fpr4, delimiter=",")
    # np.savetxt(r"E:\论文\LncRNA-miRNA\final\ROC\Tpr4.csv", tpr4, delimiter=",")
    #
    # save the average recall and precision
    # np.savetxt(r"E:\论文\LncRNA-miRNA\final\PR\mean_recall.csv", mean_recall, delimiter=",")
    # np.savetxt(r"E:\论文\LncRNA-miRNA\final\PR\mean_precision.csv", mean_precision, delimiter=",")
    #
    # np.savetxt(r"E:\论文\LncRNA-miRNA\final\PR\precision1.csv", precision1, delimiter=",")
    # np.savetxt(r"E:\论文\LncRNA-miRNA\final\PR\recall1.csv", recall1, delimiter=",")
    # np.savetxt(r"E:\论文\LncRNA-miRNA\final\PR\precision2.csv", precision2, delimiter=",")
    # np.savetxt(r"E:\论文\LncRNA-miRNA\final\PR\recall2.csv", recall2, delimiter=",")
    # np.savetxt(r"E:\论文\LncRNA-miRNA\final\PR\precision3.csv", precision3, delimiter=",")
    # np.savetxt(r"E:\论文\LncRNA-miRNA\final\PR\recall3.csv", recall3, delimiter=",")
    # np.savetxt(r"E:\论文\LncRNA-miRNA\final\PR\precision4.csv", precision4, delimiter=",")
    # np.savetxt(r"E:\论文\LncRNA-miRNA\final\PR\recall4.csv", recall4, delimiter=",")
    # np.savetxt(r"E:\论文\LncRNA-miRNA\final\PR\precision5.csv", precision5, delimiter=",")
    # np.savetxt(r"E:\论文\LncRNA-miRNA\final\PR\recall5.csv", recall5, delimiter=",")


