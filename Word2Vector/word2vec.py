from gensim.models import Word2Vec
import pandas as pd
from gensim.models.word2vec import LineSentence

sen = r'E:\论文\LncRNA-miRNA\NDALMA-main\770\dataset\Lnc_seq.txt'

sentences = LineSentence(sen)


model = Word2Vec(sentences, sg=1, size=300, window=5, min_count=1, negative=3, sample=0.001, hs=1, iter=10, batch_words=100)
print(len(model.wv.index2word))
model.save(r'E:\论文\LncRNA-miRNA\NDALMA-main\770\dataset\Lnc_seq.h5')
# model = Word2Vec.load(fname)
# model.similarity('MDKP', 'DKPY')