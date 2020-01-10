import numpy as np
from helper import tensor_prod
# This file is for draft only
operators = {"id": np.eye(2, 2),
                               "s_z": np.array([[0.5, 0], [0, -0.5]]),
                               "s_p": np.array([[0, 1], [0, 0]]),
                               "s_m": np.array([[0, 0], [1, 0]])}

key = list(operators.keys())
key.remove("id")
