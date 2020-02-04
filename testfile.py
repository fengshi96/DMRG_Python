import numpy as np
from helper import tensor_prod
from Block import Block
# This file is for draft only
left_block = Block(2, "left")
right_block = Block(2, "right")
field = [["s_x", -1]]
interaction = [["s_z", "s_z", -1]]
for i in range(len(field)):
    onsite_op = field[i][0]
    param = field[i][1]
    left_block.block_operators["block_ham"] += left_block.site_operators[onsite_op] * param
    right_block.block_operators["block_ham"] += right_block.site_operators[onsite_op] * param


super_block = left_block.glue(right_block, interaction)
evals, evecs = np.linalg.eigh(super_block.block_operators["block_ham"])
print("Energies are: ", evals)