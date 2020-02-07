# This file is for draft only
import numpy as np
from helper import tensor_prod
from Block import Block

# My code: 4 site TFIM in OBC
left_block = Block(2, "left")  # initialize left block as a spin-1/2 site
right_block = Block(2, "right")  # initialize right block as a spin-1/2 site
field = [["s_x", -1]]
interaction = [["s_z", "s_z", -1]]

for i in range(len(field)):  # define single site hamiltonian if field != None
    onsite_op = field[i][0]
    param = field[i][1]
    left_block.block_operators["block_ham"] += left_block.site_operators[onsite_op] * param
    right_block.block_operators["block_ham"] += right_block.site_operators[onsite_op] * param

left_block.grow(interaction, field)  # left block grows from single site to 2-site block
right_block.grow(interaction, field)  # right block grows from single site to 2-site block

super_block = left_block.glue(right_block, interaction)  # make 4-site super block by connecting left and right block
evals, evecs = np.linalg.eigh(super_block.block_operators["block_ham"])  # diagonalize the super block
print("(my code) Energies are: ", evals)

# Exact: 4-site TFIM in OBC
sx = np.array([[0, 0.5], [0.5, 0]])
sz = np.array([[0.5, 0], [0, -0.5]])
id = np.eye(2, 2)

ham_connector = np.zeros((16, 16))
ham_onsite = np.zeros((16, 16))

ham_connector += tensor_prod(tensor_prod(tensor_prod(sz, sz), id), id)
ham_connector += tensor_prod(tensor_prod(tensor_prod(id, sz), sz), id)
ham_connector += tensor_prod(tensor_prod(tensor_prod(id, id), sz), sz)

ham_onsite += tensor_prod(tensor_prod(tensor_prod(sx, id), id), id)
ham_onsite += tensor_prod(tensor_prod(tensor_prod(id, sx), id), id)
ham_onsite += tensor_prod(tensor_prod(tensor_prod(id, id), sx), id)
ham_onsite += tensor_prod(tensor_prod(tensor_prod(id, id), id), sx)

ham = -ham_connector - ham_onsite
evals, evecs = np.linalg.eigh(ham)
print("(exact) Energies are: ", evals)

