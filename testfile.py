import numpy as np
from helper import tensor_prod
block_operators = {"id": np.eye(4, 4),
                        "block_ham": np.zeros((4, 4)),
                        "s_z": tensor_prod(np.array([[0.5, 0], [0, -0.5]]),
                                           np.eye(int(4 / 2), int(4 / 2))),
                        "s_p": tensor_prod(np.array([[0, 1], [0, 0]]),
                                           np.eye(int(4 / 2), int(4 / 2))),
                        "s_m": tensor_prod(np.array([[0, 0], [1, 0]]),
                                           np.eye(int(4 / 2), int(4 / 2)))}

print("id" in block_operators.keys())
i = 1
for op in block_operators.keys():
    print(block_operators[op])
    i += 1

print(i)