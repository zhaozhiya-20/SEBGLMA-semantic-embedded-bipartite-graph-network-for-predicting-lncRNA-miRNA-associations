import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)  # 构建第一层 GCN
        self.gc2 = GraphConvolution(nhid, nclass)  # 构建第二层 GCN
        self.dropout = dropout

    def forward(self, x, adj):
        ZZ = self.gc1(x, adj)
        x = F.relu(ZZ)
        #
        ZZ1 = ZZ.detach().numpy()
        # np.savetxt(r"E:\论文\LncRNA-miRNA\NDALMA-main\770\make_dataset\final\feature_new_150.csv", ZZ1, delimiter=",")
        # np.savetxt(r"E:\论文\LncRNA-miRNA\case_study\final\feature_gcn_150_3.csv", ZZ1, delimiter=",")

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)
