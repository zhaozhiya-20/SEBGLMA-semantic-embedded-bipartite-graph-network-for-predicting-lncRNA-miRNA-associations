import pandas as pd
import numpy as np
import random

# all = pd.read_csv(r"E:\论文\LncRNA-miRNA\CS2\negtive.csv", header=None)
all = pd.read_csv(r"E:\论文\Agithub\CS\hsa-miR-497-5P\neg_id.csv", header=None)
all = np.array(all)
print(all.shape)

index = [i for i in range(102660)]
random.shuffle(index)
input_new = all[index]

print(input_new[0])

negtive = np.zeros((7141, 2))
for j in range(7141):
    negtive[j, 0] = input_new[j, 0]
    negtive[j, 1] = input_new[j, 1]
np.savetxt(r"E:\论文\Agithub\CS\hsa-miR-497-5P\neg.csv", negtive, delimiter=",")