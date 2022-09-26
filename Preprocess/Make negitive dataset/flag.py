import pandas as pd
import numpy as np

positive = pd.read_csv(r"E:\论文\LncRNA-miRNA\CS2\Pos_id.csv", header=None)
all = pd.read_csv(r"E:\论文\LncRNA-miRNA\case_study\final\all.csv", header=None)

positive = np.array(positive)
all = np.array(all)
all = all.astype(int)
print(positive.shape)
print(all.shape)

flag = [0 for index in range(110197)]
# neg = np.zeros((250000, 2))
# z = 0
for i in range(110197):
    for j in range(7188):
        if positive[j, 0] == all[i, 0] and positive[j, 1] == all[i, 1]:
            flag[i] = 1
            # z = z+1

np.savetxt(r"E:\论文\LncRNA-miRNA\CS2\flag.csv", flag, delimiter=",")