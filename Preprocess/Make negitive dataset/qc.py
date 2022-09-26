import pandas as pd
import numpy as np

all = pd.read_csv(r"E:\论文\Agithub\CS\hsa-miR-497-5P\except_497.csv", header=None)
flag = pd.read_csv(r"E:\论文\Agithub\CS\hsa-miR-497-5P\flag2.csv", header=None)
all = np.array(all)
flag = np.array(flag)

print(all.shape)
print(flag.shape)

neg = np.zeros((102660, 2))
z = 0

for i in range(109778):
    if flag[i, 0] == 0:
        neg[z, 0] = all[i, 0]
        neg[z, 1] = all[i, 1]
        z = z+1
print(neg.shape)
np.savetxt(r"E:\论文\Agithub\CS\hsa-miR-497-5P\neg_id.csv", neg, delimiter=",")