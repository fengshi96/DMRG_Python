import sys
from Warmup import *
from Wavefunction import Wavefunction
from helper import Logger

# for exporting the logfile
sys.stdout = Logger()

# Warm up by infinite size DMRG
num_sites = 32
dmax = 22
interaction = [["s_p", "s_m", 0.5], ["s_m", "s_p", 0.5], ["s_z", "s_z", 1]]
left_block, right_block, storage = warmup(num_sites, dmax, interaction)

num_sweeps = 3
half_sweeps = 0
super_block = left_block.glue(right_block, interaction)
print("(fDMRG)left_block.dim before sweep is ", left_block.dim)

while half_sweeps < 2 * num_sweeps:
    print("!!left storage, ", storage.left_dim)
    print("!!right storage, ", storage.right_dim)
    rsize_max = int(right_block.num_sites)
    # left to right
    for rsize in range(rsize_max - 1, 0, -1):
        print("rsize = ", rsize)
        sweep("left", left_block, right_block, interaction, storage)
        print("left_block.dim (The current growing side's dim) = ", left_block.dim)
        super_block = left_block.glue(right_block, interaction)

        left_dim = left_block.dim
        right_dim = right_block.dim
        evals, evecs = np.linalg.eigh(super_block.block_operators["block_ham"])  # diagonalize the super_block
        # bipartition of the ground state wavefunction
        wf_gs = Wavefunction(left_dim, right_dim)
        wf_gs.as_matrix = np.reshape(evecs[:, 0], (left_dim, right_dim))

        if left_dim > dmax:
            print("Truncation")
            left_rdm = wf_gs.rdm("right")
            left_evals, left_evecs = np.linalg.eigh(left_rdm)
            left_evecs_truncated = left_evecs[:, -dmax:]
            left_evals_truncated = left_evals[-dmax:]
            truncation_error = 1.0 - sum(left_evals_truncated)
            print("truncation_error = ", truncation_error)

            # Rotate all block_operators in left block if needed
            left_block.truncate(left_evecs_truncated)
            left_block.dim = dmax

        if right_dim > dmax:
            print("Truncation")
            right_rdm = wf_gs.rdm("left")
            right_evals, right_evecs = np.linalg.eigh(right_rdm)
            right_evecs_truncated = right_evecs[:, -dmax:]
            right_evals_truncated = right_evals[-dmax:]

            # Rotate all block_operators in right block
            right_block.truncate(right_evecs_truncated)
            right_block.dim = dmax

            truncation_error = 1.0 - sum(right_evals_truncated)
            print("truncation_error = ", truncation_error)

        # Take a picture
        storage.snapshot(left_block, right_block, "left")

    storage.right_operators = []
    storage.right_dim = []
    half_sweeps += 1
    print("--------------This is the end of " + str(half_sweeps) + "th Half-Sweep (L2R)--------------")

    if half_sweeps == 2 * num_sweeps - 1:
        # the last half sweep
        lsize_max = int(num_sites / 2)
    else:
        # right to left
        lsize_max = int(left_block.num_sites)

    for lsize in range(lsize_max - 1, 0, -1):
        print("lsize = ", lsize)
        print("left_block, right_block = ", left_block.dim, right_block.dim)
        sweep("right", left_block, right_block, interaction, storage)
        print("left_block, right_block (After dfmrg) = ", left_block.dim, right_block.dim)
        super_block = left_block.glue(right_block, interaction)
        left_dim = left_block.dim
        right_dim = right_block.dim
        evals, evecs = np.linalg.eigh(super_block.block_operators["block_ham"])  # diagonalize the super_block

        # bipartition of the ground state wavefunction
        wf_gs = Wavefunction(left_dim, right_dim)
        wf_gs.as_matrix = np.reshape(evecs[:, 0], (left_dim, right_dim))

        if left_dim > dmax:
            print("Truncation")
            left_rdm = wf_gs.rdm("right")
            left_evals, left_evecs = np.linalg.eigh(left_rdm)
            left_evecs_truncated = left_evecs[:, -dmax:]
            left_evals_truncated = left_evals[-dmax:]
            truncation_error = 1.0 - sum(left_evals_truncated)
            print("truncation_error = ", truncation_error)

            # Rotate all block_operators in left block if needed
            left_block.truncate(left_evecs_truncated)
            left_block.dim = dmax

        if right_dim > dmax:
            print("Truncation")
            right_rdm = wf_gs.rdm("left")
            right_evals, right_evecs = np.linalg.eigh(right_rdm)
            right_evecs_truncated = right_evecs[:, -dmax:]
            right_evals_truncated = right_evals[-dmax:]

            # Rotate all block_operators in right block
            right_block.truncate(right_evecs_truncated)
            right_block.dim = dmax

            truncation_error = 1.0 - sum(right_evals_truncated)
            print("truncation_error = ", truncation_error)
        # Take a picture
        storage.snapshot(left_block, right_block, "right")

    storage.left_operators = []
    storage.left_dim = []
    half_sweeps += 1
    print("--------------This is the end of " + str(half_sweeps) + "th Half-Sweep (R2L)--------------")

evals, evecs = np.linalg.eigh(super_block.block_operators["block_ham"])  # diagonalize the final super_block
print("Eigen values are: ", evals)
print("Ground state energy = ", min(evals))
print("number of left sites = ", left_block.num_sites)
print("number of right sites = ", right_block.num_sites)
