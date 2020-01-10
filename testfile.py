import numpy as np
from helper import tensor_prod
# This file is for draft only
a = np.array([1,2,4,3])
index_ascend = np.argsort(a)
print(index_ascend)
print(a[index_ascend])