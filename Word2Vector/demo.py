import pandas as pd
import xlrd
import numpy as np

import pandas as pd
import xlrd
import numpy as np

path = r'E:\论文\分类器\Word2Vec\dataset\拟南芥\negative\negative_a.xlsx'
data = xlrd.open_workbook(path)
table = data.sheets()[0]
nrows = table.nrows # 行
ncols = table.ncols  # 列
print(nrows, ncols)

#将每一行内容一次存入列表c中
r = []
for i in range(0,nrows):
    # 某一行不包含序列的内容
    rowValues= table.row_values(i)
    r.append(rowValues)

print(type(r))
print(len(r))
print(r[0])
print(type(r[0]))
# print(len(r[0]))
def Kmers_funct(seq, size):
    return [seq[x:x+size].upper for x in range(len(seq) - size + 1)]

# a = 'ABCDEFGHIGKLKJHHG'
# kmer = Kmers_funct(a, size=7)
#
# kmer = ''.join(kmer)
# print(kmer)
# print(type(p))
# a = ''.join(r[0])
# kmer = Kmers_funct(a, size=7)
# print(kmer)

# full_data = []
# for i in range(2):
#     a = ''.join(r[i])
#     kmer = Kmers_funct(a, size=4)
#     # print(kmer)
#     # print(kmer[0]())
#     # print(len(kmer))
#     # print(type(kmer[0]()))
#     new = []
#     for i in range(len(kmer)):
#         new.append(kmer[i]())
#     new = ''.join(new)
#     full_data.append(new)
# print(len(full_data))
# print(full_data[0])
#
# np.savetxt(r'E:\论文\分类器\Word2Vec\dataset\拟南芥\negative\negative_a3.csv',
#          full_data, fmt='%s', delimiter=",")
z = ['qwe', 'asd','qwe']
print(' '.join(z))
