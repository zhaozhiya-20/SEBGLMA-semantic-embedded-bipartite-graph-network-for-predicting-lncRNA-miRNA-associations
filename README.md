# SEBGLMA: semantic embedded bipartite graph network for predicting lncRNA-miRNA associations
This project includes the codes of SEBGLMA and comparison models. It also provides complete data of benchmark data set and case studies including interacting pairs, RNA indexes, adjacency matrix, and negative data sets.

1. We provide the software of K-mer, word2vec, GIP, GCN, and RoF to assist constructing the proposed model.
2. We provide the codes of LGBM, SVM and RF to help constructing the comparison, the code of ablation experiments is also given.
3. In benchmark data set, we provide the positive data set containing 4966 interacting lncRNA-miRNA pairs, 4966 non-interacting pairs, and adjacency matrix without self-similarity. Within case studies, we provide the data set rescreening from lncRNASNP2. Specifically, we divide the case related data from the complete data set, and they are regarded as train dataset and test dataset, respectively.
