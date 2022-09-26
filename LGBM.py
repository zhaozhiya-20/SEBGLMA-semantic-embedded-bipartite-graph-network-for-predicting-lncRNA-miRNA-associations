from sklearn.metrics import accuracy_score, precision_score, f1_score
from lightgbm import LGBMClassifier
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from numpy import interp
from sklearn.metrics import precision_recall_curve

mean_fpr = np.linspace(0, 1, 100)

lgbm = LGBMClassifier(num_leaves=60, learning_rate=0.05, n_estimators=40)

input_path = r'E:\论文\LncRNA-miRNA\NDALMA-main\770\make_dataset\final\feature_150\N.xlsx'
input_path1 = r'E:\论文\LncRNA-miRNA\NDALMA-main\770\make_dataset\final\feature_150\P.xlsx'
input_data = pd.read_excel(input_path, header=None)
input_data1 = pd.read_excel(input_path1, header=None)

input0 = np.array(input_data)
input1 = np.array(input_data1)
input = np.vstack((input0, input1))

index = [i for i in range(9932)]
random.shuffle(index)
input_new = input[index]

# y = input[10:20,0]
# y_new = input_new[10:20,0]


X = input_new[:, 1:301]
Y = input_new[:, 0]

KF = KFold(n_splits=5)

X_train = []
Y_train = []
X_test = []
Y_test = []

Real_data = []
Predict_data = []

for train_index, test_index in KF.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    x_train, x_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    X_train.append(x_train)
    Y_train.append(y_train)
    X_test.append(x_test)
    Y_test.append(y_test)

    lgbm.fit(x_train, y_train)
    y_pre = lgbm.predict(x_test)
    y_pre1 = lgbm.predict_proba(x_test)
    Real_data.append(y_test)

    Predict_data.append(y_pre1[:, 1])

    f1 = f1_score(y_test, y_pre, average='micro')
    print("the f1 score: %.2f"%f1)
    C2 = confusion_matrix(y_test, y_pre, labels=[0, 1])
    print(C2)
    print(C2.ravel())
    C = C2.ravel()
    MCC = (C[0] * C[3] - C[1] * C[2]) / ((C[3] + C[1]) * (C[3] + C[2]) * (C[0] + C[1]) * (C[0] + C[2])) ** 0.5
    Sen = C[3] / (C[3] + C[2])
    Acc = (C[3] + C[0]) / (C[3] + C[0] + C[1] + C[2])
    Pre = C[3] / (C[1] + C[3])
    Spec = C[0] / (C[0] + C[1])
    print("******************************************")
    print("Acc:", Acc)
    print("Pre:", Pre)
    print("Sen:", Sen)
    print("Spec:", Spec)
    print("MCC:", MCC)

    # print(classification_report(y, final_prediction, digits=4))  # Pre、Sen(recall)
print("******************************************")

tprs = []

# y_pred_gbc = lgbm.predict_proba(X_test[0])

fpr, tpr, threshold = roc_curve(Real_data[0], Predict_data[0])
tprs.append(interp(mean_fpr, fpr, tpr))
fpr1, tpr1, threshold1 = roc_curve(Real_data[1], Predict_data[1])
tprs.append(interp(mean_fpr, fpr1, tpr1))
fpr2, tpr2, threshold2 = roc_curve(Real_data[2], Predict_data[2])
tprs.append(interp(mean_fpr, fpr2, tpr2))
fpr3, tpr3, threshold3 = roc_curve(Real_data[3], Predict_data[3])
tprs.append(interp(mean_fpr, fpr3, tpr3))
fpr4, tpr4, threshold4 = roc_curve(Real_data[4], Predict_data[4])
tprs.append(interp(mean_fpr, fpr4, tpr4))

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (area=%0.2f)' % mean_auc, lw=2, alpha=0.8)

roc_auc = auc(fpr, tpr)
print(roc_auc)
roc_auc1 = auc(fpr1, tpr1)
print(roc_auc1)
roc_auc2 = auc(fpr2, tpr2)
print(roc_auc2)
roc_auc3 = auc(fpr3, tpr3)
print(roc_auc3)
roc_auc4 = auc(fpr4, tpr4)
print(roc_auc4)
average_auc = (roc_auc + roc_auc1 + roc_auc2 + roc_auc3 + roc_auc4) / 5
lw = 4
fig = plt.figure(figsize=(10, 10))
# plt.plot(fpr, tpr, color='darkorange', lw=lw, label='AUC_Fold1  (area = %0.3f)' % roc_auc)
plt.plot(fpr, tpr, color='black', lw=lw, label='1st fold = %0.4f' % roc_auc)
plt.plot(fpr1, tpr1, color='blue', lw=lw, label='2nd fold = %0.4f' % roc_auc1)
plt.plot(fpr2, tpr2, color='red', lw=lw, label='3rd fold = %0.4f' % roc_auc2)
plt.plot(fpr3, tpr3, color='yellow', lw=lw, label='4th fold = %0.4f' % roc_auc3)
plt.plot(fpr4, tpr4, color='green', lw=lw, label='5th fold = %0.4f' % roc_auc4)
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.text(0.22, 0.05, "Average AUC = %0.4f" % average_auc, size=34, alpha=1)
plt.xlabel('1-Specificity', fontsize=30)
plt.ylabel('Sensitivity', fontsize=30)
# plt.title('Receiver operating characteristic example')
plt.legend(loc="right", fontsize=26)

ax = plt.gca()
ax.spines['bottom'].set_linewidth(2)
ax.spines['left'].set_linewidth(2)
ax.spines['right'].set_linewidth(2)
ax.spines['top'].set_linewidth(2)

plt.tick_params(labelsize=24)
plt.show()

fig.savefig(r'E:\论文\LncRNA-miRNA\result\LGBM\LGBM_ROC.tif', dpi=600, format='tif')
np.savetxt(r"E:\论文\LncRNA-miRNA\result\LGBM\mean_Fpr.csv", mean_fpr, delimiter=",")
np.savetxt(r"E:\论文\LncRNA-miRNA\result\LGBM\mean_Tpr.csv", mean_tpr, delimiter=",")
# np.savetxt(r"E:\论文\LncRNA-miRNA\result\LGBM\Fpr.csv", mean_fpr, delimiter=",")
# np.savetxt(r"E:\论文\LncRNA-miRNA\result\LGBM\Tpr.csv", mean_tpr, delimiter=",")

mean_recall = np.linspace(1, 0, 100)  # Linear interpolation method
precisions = []
# Prepare for PR curve
precision1, recall1, thresholds = precision_recall_curve(Real_data[0], Predict_data[0], pos_label=None,
                                                         sample_weight=None)
precision1 = np.matrix.tolist(precision1)  # 逆序recall和precision
recall1 = np.matrix.tolist(recall1)
precision1.reverse()
recall1.reverse()
precisions.append(interp(mean_recall, recall1, precision1))

precision2, recall2, thresholds2 = precision_recall_curve(Real_data[1], Predict_data[1], pos_label=None,
                                                          sample_weight=None)
precision2 = np.matrix.tolist(precision2)
recall2 = np.matrix.tolist(recall2)
precision2.reverse()
recall2.reverse()
precisions.append(interp(mean_recall, recall2, precision2))

precision3, recall3, thresholds3 = precision_recall_curve(Real_data[2], Predict_data[2], pos_label=None,
                                                          sample_weight=None)
precision3 = np.matrix.tolist(precision3)  # 逆序recall和precision
recall3 = np.matrix.tolist(recall3)
precision3.reverse()
recall3.reverse()
precisions.append(interp(mean_recall, recall3, precision3))

precision4, recall4, thresholds4 = precision_recall_curve(Real_data[3], Predict_data[3], pos_label=None,
                                                          sample_weight=None)
precision4 = np.matrix.tolist(precision4)  # 逆序recall和precision
recall4 = np.matrix.tolist(recall4)
precision4.reverse()
recall4.reverse()
precisions.append(interp(mean_recall, recall4, precision4))

precision5, recall5, thresholds5 = precision_recall_curve(Real_data[4], Predict_data[4], pos_label=None,
                                                          sample_weight=None)
precision5 = np.matrix.tolist(precision5)  # 逆序recall和precision
recall5 = np.matrix.tolist(recall5)
precision5.reverse()
recall5.reverse()
precisions.append(interp(mean_recall, recall5, precision5))


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
fig2.savefig(r'E:\论文\LncRNA-miRNA\result\LGBM\LGBM_PR.tif', dpi=600, format='tif')
# fig.savefig(r'E:\论文\DTI\FLTP_V2\NEW\SVM\TIFF\EN_SVM.tif', dpi=600, format='tif')

# np.savetxt(r"E:\论文\LncRNA-miRNA\result\LGBM\mean_Fpr.csv", mean_fpr, delimiter=",")
# np.savetxt(r"E:\论文\LncRNA-miRNA\result\LGBM\mean_Tpr.csv", mean_tpr, delimiter=",")


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
np.savetxt(r"E:\论文\LncRNA-miRNA\result\LGBM\mean_recall.csv", mean_recall, delimiter=",")
np.savetxt(r"E:\论文\LncRNA-miRNA\result\LGBM\mean_precision.csv", mean_precision, delimiter=",")