from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.decomposition import PCA
from pygcn.utils import load_data, accuracy
from pygcn.models import GCN
import pandas as pd
import numpy as np
import scipy.sparse as sp
import torch

# Training settings
# 建立解析对象，加上--将之变为可选参数
parser = argparse.ArgumentParser()
# 增加属性
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=41, help='Random seed.')
parser.add_argument('--epochs', type=int, default=500,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=150,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

# 将所有属性返回到args子类实例
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
# MAC: option + command + <-


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))  # 矩阵行求和
    r_inv = np.power(rowsum, -1).flatten()  # 求和的-1次方
    r_inv[np.isinf(r_inv)] = 0.   # 如果是inf，转换成0
    r_mat_inv = sp.diags(r_inv)  # 构造对角戏矩阵
    mx = r_mat_inv.dot(mx)  # 构造D-1*A，非对称方式，简化方式
    return mx

# Load data
def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

pca = PCA(n_components=200)

adj = pd.read_csv(r"E:\论文\LncRNA-miRNA\case_study\adj\adj.csv", header=None)
# idx_feature_label = pd.read_csv(r"E:\论文\LncRNA-miRNA\NDALMA-main\770\make_dataset\final\feature.csv", header=None)
idx_feature_label = pd.read_excel(r"E:\论文\LncRNA-miRNA\case_study\data\id_feature_label.xlsx", header=None)
idx_feature_label = np.array(idx_feature_label)

ifl = idx_feature_label
# index = [i for i in range(1045)]
# random.shuffle(index)
# ifl = idx_feature_label[index]

adj = np.array(adj)
I = np.identity(682)
adj = normalize(adj)

ifl = np.array(ifl)

features = sp.csr_matrix(ifl[:, 1:-1], dtype=np.float32)  # 取特征feature
labels = encode_onehot(ifl[:, -1])  # one-hot label


# 训练，验证，测试的样本
idx_train = range(0, 682)
idx_val = range(200, 500)
idx_test = range(0, 682)
features = normalize(features)
features = torch.FloatTensor(np.array(features.todense()))
labels = torch.LongTensor(np.where(labels)[1])
adj = torch.FloatTensor(adj)

idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)



# adj, features, labels, idx_train, idx_val, idx_test = load_data()

# Model and optimizer,构造GCN，初始化参数。两层GCN
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=2,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()  # GraphConvolution forward，梯度值清零
    output = model(features, adj)   # 运行模型，输入参数 (features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train]) # 先log_softmax(),然后nll_loss就相当于一个cross_entropy()

    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()  # 反向传播计算梯度值
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))


def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))


# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()
