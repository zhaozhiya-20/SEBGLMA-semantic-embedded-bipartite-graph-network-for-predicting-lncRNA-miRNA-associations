import pandas as pd
import numpy as np
import scipy.sparse as sp
import torch
import random

def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))  # 矩阵行求和
    r_inv = np.power(rowsum, -1).flatten()  # 求和的-1次方
    r_inv[np.isinf(r_inv)] = 0.   # 如果是inf，转换成0
    r_mat_inv = sp.diags(r_inv)  # 构造对角戏矩阵
    mx = r_mat_inv.dot(mx)  # 构造D-1*A，非对称方式，简化方式
    return mx

adj = pd.read_csv(r"E:\论文\LncRNA-miRNA\NDALMA-main\770\make_dataset\final\adj.csv", header=None)
idx_feature_label = pd.read_csv(r"E:\论文\LncRNA-miRNA\NDALMA-main\770\make_dataset\final\feature.csv", header=None)
adj = np.array(adj)
idx_feature_label = np.array(idx_feature_label)

index = [i for i in range(1045)]
random.shuffle(index)
ifl = idx_feature_label[index]

features = sp.csr_matrix(ifl[:, 1:-1], dtype=np.float32)  # 取特征feature
labels = encode_onehot(ifl[:, -1])  # one-hot label

# 训练，验证，测试的样本
idx_train = range(200)
idx_val = range(200, 500)
idx_test = range(500, 1045)

I = np.identity(1045)
adj = normalize(adj+I)
normalize(features)
features = torch.FloatTensor(np.array(features.todense()))
labels = torch.LongTensor(np.where(labels)[1])
adj = torch.FloatTensor(adj)

idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)



# print(0)