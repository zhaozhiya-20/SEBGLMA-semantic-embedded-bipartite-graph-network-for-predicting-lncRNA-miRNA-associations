import pandas as pd
import numpy as np

# input data

pos = pd.read_csv(r"E:\论文\LncRNA-miRNA\CS2\Pos_id.csv", header=None)
neg = pd.read_csv(r"E:\论文\LncRNA-miRNA\CS2\Neg_train.csv", header=None)
# features = pd.read_csv(r"E:\论文\LncRNA-miRNA\NDALMA-main\770\make_dataset\final\f_np.csv", header=None)
# features = pd.read_csv(r"E:\论文\LncRNA-miRNA\NDALMA-main\770\make_dataset\final\feature_new_150.csv", header=None)
# features = pd.read_csv(r"E:\论文\分类器\Word2Vec_LNCRNA\feature_500\feature_500_150.csv", header=None)
features = pd.read_csv(r"E:\论文\LncRNA-miRNA\case_study\final\feature_gcn_150.csv", header=None)
neg = np.array(neg)
pos = np.array(pos)
features = np.array(features)
# f = features.values
# np.savetxt(r"E:\论文\LncRNA-miRNA\NDALMA-main\770\make_dataset\final\f_np.csv", f, delimiter=",")
PA = []
PB = []
NA = []
NB = []
P = []
N = []
print(pos.shape)
print(neg.shape)
print(features.shape)
print(type(neg))
# print(pos[0, 0])
# print(type(features))

for i in range(7188):
    PA.append(features[pos[i, 0]])
    PB.append(features[pos[i, 1]])
    NA.append(features[neg[i, 0]])
    NB.append(features[neg[i, 1]])
# print(len(PA))
# # print(type(PA))
# np.savetxt(r"E:\论文\LncRNA-miRNA\NDALMA-main\770\make_dataset\final\PA_150.csv", PA, delimiter=",")
# np.savetxt(r"E:\论文\LncRNA-miRNA\NDALMA-main\770\make_dataset\final\PB_150.csv", PB, delimiter=",")
# np.savetxt(r"E:\论文\LncRNA-miRNA\NDALMA-main\770\make_dataset\final\NA_150.csv", NA, delimiter=",")
# np.savetxt(r"E:\论文\LncRNA-miRNA\NDALMA-main\770\make_dataset\final\NB_150.csv", NB, delimiter=",")
# np.savetxt(r"E:\论文\分类器\Word2Vec_LNCRNA\feature_500\PN\PA_500_150.csv", PA, delimiter=",")
# np.savetxt(r"E:\论文\分类器\Word2Vec_LNCRNA\feature_500\PN\PB_500_150.csv", PB, delimiter=",")
# np.savetxt(r"E:\论文\分类器\Word2Vec_LNCRNA\feature_500\PN\NA_500_150.csv", NA, delimiter=",")
# np.savetxt(r"E:\论文\分类器\Word2Vec_LNCRNA\feature_500\PN\\NB_500_150.csv", NB, delimiter=",")
np.savetxt(r"E:\论文\LncRNA-miRNA\CS2\PN\PA.csv", PA, delimiter=",")
np.savetxt(r"E:\论文\LncRNA-miRNA\CS2\PN\PB.csv", PB, delimiter=",")
np.savetxt(r"E:\论文\LncRNA-miRNA\CS2\PN\NA.csv", NA, delimiter=",")
np.savetxt(r"E:\论文\LncRNA-miRNA\CS2\PN\NB.csv", NB, delimiter=",")



