from gensim.models import Word2Vec
import numpy as np
import pandas as pd
from gensim.models.word2vec import LineSentence
def normalization(data):
    minVals = data.min()
    maxVals = data.max()
    ranges = maxVals - minVals
    normData = np.zeros(np.shape(data))
    m = data.shape[0]
    normData = data - np.tile(minVals, (m, 1))
    normData = normData/np.tile(ranges, (m, 1))
    return normData

def list_add(a,b):
    c = []
    for i in range(len(a)):
        c.append(a[i]+b[i])
    return c

model = Word2Vec.load(r'E:\论文\LncRNA-miRNA\NDALMA-main\770\dataset\Lnc_seq.h5')
# print(model.similarity('MDKP', 'DKPY'))
# print(model['GUAG'])
# print(model['UAAG'])
# A = model["UAAG"]
# B = normalization(A)
# print(len(B))
sen = r'E:\论文\LncRNA-miRNA\NDALMA-main\770\dataset\Lnc_seq.txt'

feature = []
# for line in open(sen):
#
#     line.rstrip('\n')
#     print(line)

mir_sen = []
with open(sen) as f:
    lines = f.read().splitlines()
    # print(lines)
    for i in range(len(lines)):
        miRNA = lines[i].split(" ")
        miRNA = list(miRNA)
        # print(miRNA)
        mir_sen.append(miRNA)
        # print(mir_sen)

    miRNA_feature = []
    for i in range(len(mir_sen)):
        feature_sum = []
        for ii in range(300):
            feature_sum.append(0)
        for j in range(len(mir_sen[i])):
            word_feature = np.array(model[mir_sen[i][j]])
            feature_sum = np.array(feature_sum)
            feature_sum = list(np.add(word_feature, feature_sum))
        # feature_sum = feature_sum[0]
            # print(feature_sum)
            # for jj in range(100):
            #     feature_sum[jj] = normalization(model[mir_sen[i][j]])[jj]+feature_sum[jj]
        miRNA_feature.append(feature_sum)
    print(len(miRNA_feature))

    print(miRNA_feature[0])
    np.savetxt(r"E:\论文\LncRNA-miRNA\NDALMA-main\770\dataset\Lnc_word_feature.csv", miRNA_feature, delimiter=",")





