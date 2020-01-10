import numpy as np
from helper import tensor_prod
# This file is for draft only
t = tensor_prod(np.eye(2,2), np.array([[0,0],[0,0]]))
g = tensor_prod(np.array([[0.5,0],[0,-0.5]]),np.eye(1,1))
print(t)
print(g)
print(np.eye(1,1))