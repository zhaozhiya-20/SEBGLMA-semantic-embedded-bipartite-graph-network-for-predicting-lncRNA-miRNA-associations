import pandas as pd
import numpy as np
import random

all = pd.read_csv(r"E:\论文\Agithub\CS\Neg_all.csv", header=None)

all = np.array(all)

index = [i for i in range(102784)]
random.shuffle(index)
input_new = all[index]

print(input_new[0])

negtive = np.zeros((7150, 2))
for j in range(7150):
    negtive[j, 0] = input_new[j, 0]
    negtive[j, 1] = input_new[j, 1]
np.savetxt(r"E:\论文\Agithub\CS\neg.csv", negtive, delimiter=",")