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
# print(model['UAG'])
# print(model['UAU'])
print(model['AGCC'])
print(model.similarity('AGCC', 'GCCG'))