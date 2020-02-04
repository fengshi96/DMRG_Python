import numpy as np
from Block import Block
from Memory import Memory
from helper import plot
from Wavefunction import Wavefunction


def warmup(num_sites, dmax, interaction, field=[]):
    """
    Or the infinite size DMRG algorithm - the preparation for finite size DMRG
    Parameters:
        num_sites: the total number of sites of the system.
        dmax: the maximal number of states to keep
        interaction: interaction matrix between sites
    """
    # initialize 2 spins on left and right.
    left_block = Block(2, "left")
    right_block = Block(2, "right")

    # define single site Hamiltonian if field is present; otherwise single site Hamiltonian is zero
    for i in range(len(field)):
        onsite_op = field[i][0]
        param = field[i][1]
        left_block.block_operators["block_ham"] += left_block.site_operators[onsite_op] * param
        right_block.block_operators["block_ham"] += right_block.site_operators[onsite_op] * param

    block_sites = num_sites / 2  # number of sites per sub block
    storage = Memory()

    iteration = 1
    while True:
        storage.snapshot(left_block, right_block, None)
        plot(left_block.num_sites, right_block.num_sites, None) # show geometry
        if iteration >= block_sites:
            break

        print("(iDMRG)"+str(iteration) + "th iteration")
        left_block.grow(interaction, field)
        right_block.grow(interaction, field)
        super_block = left_block.glue(right_block, interaction)
        print("(iDMRG) truncated superblock: ", super_block.block_operators["block_ham"].shape,
              left_block.block_operators["id"].shape, right_block.block_operators["id"].shape)

        if left_block.dim >= dmax:
            evals, evecs = np.linalg.eigh(super_block.block_operators["block_ham"])  # diagonalize the super_block

            # bipartition of the ground state wavefunction
            wf_gs = Wavefunction(left_block.dim, right_block.dim)
            wf_gs.as_matrix = np.reshape(evecs[:, 0], (left_block.dim, right_block.dim))

            # Rotate all block_operators in left block
            left_block.truncate(wf_gs, dmax, "left")
            left_block.dim = dmax
            # Rotate all block_operators in right block
            right_block.truncate(wf_gs, dmax, "right")
            right_block.dim = dmax

        iteration += 1

    print(super_block.dim)
    evals, evecs = np.linalg.eigh(super_block.block_operators["block_ham"])
    print("(iDMRG)Eigen values are: ", evals)
    print("(iDMRG)Ground state energy = ", min(evals))
    print("(iDMRG)number of left sites = ", left_block.num_sites)
    print("--------------This is the end of iDMRG --------------")
    return left_block, right_block, storage


def sweep(growing_side, left_block, right_block, interaction, field, storage):
    """
    Apply one step in sweep: grow and shrink
    """

    if growing_side not in ('left', 'right'):
        raise TypeError("(Sweep) Growing side must be left or right.")
    print("(Sweep) The growing side is ", growing_side)
    if growing_side == "left":
        shrinking_block = right_block
        growing_side = left_block
        shrinking_block.block_operators = storage.right_operators[shrinking_block.num_sites - 2].copy()
        shrinking_block.dim = storage.right_dim[shrinking_block.num_sites - 2]
    else:
        shrinking_block = left_block
        growing_side = right_block
        shrinking_block.block_operators = storage.left_operators[shrinking_block.num_sites - 2].copy()
        shrinking_block.dim = storage.left_dim[shrinking_block.num_sites - 2]

    # add one more site to the left block
    growing_side.grow(interaction, field)
    shrinking_block.num_sites -= 1
    print("(After one step in Sweep) shrinking_block.dim = ", shrinking_block.dim)
    print("(After one step in Sweep) shrinking_block.num_sites = ", shrinking_block.num_sites)

    return 0
