import sys
from Warmup import *
from Wavefunction import Wavefunction
from helper import Logger, plot

# for exporting the logfile
sys.stdout = Logger()

num_sites = 16 # total number of sites in the chain
dmax = 22 # maximal number of states to keep
interaction = [["s_p", "s_m", 0.5], ["s_m", "s_p", 0.5], ["s_z", "s_z", 1]] # define isotropic Heisenberg interaction

# Warm up by infinite size DMRG
left_block, right_block, storage = warmup(num_sites, dmax, interaction)

num_sweeps = 3 # define the total number of sweeps
half_sweeps = 0 # for iteration, begin at 0
super_block = left_block.glue(right_block, interaction)
print("(fDMRG)left_block.dim before sweep is ", left_block.dim)

while half_sweeps < 2 * num_sweeps:
    rsize_max = int(right_block.num_sites)
    # left to right
    for rsize in range(rsize_max - 1, 0, -1):
        print("[storage] left_dim in  storage, ", storage.left_dim)
        print("[storage] right_dim in storage, ", storage.right_dim)
        print("rsize = ", rsize)
        sweep("left", left_block, right_block, interaction, storage)
        plot(left_block.num_sites, right_block.num_sites, "left")  # show geometry
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

    storage.erase("right")
    half_sweeps += 1
    print("--------------This is the end of " + str(half_sweeps) + "th Half-Sweep (L2R)--------------")

    if half_sweeps == 2 * num_sweeps - 1:
        # manually gauge steps for the last half sweep
        lsize_max = int(num_sites / 2)
    else:
        # right to left sweep
        lsize_max = int(left_block.num_sites)

    for lsize in range(lsize_max - 1, 0, -1):
        print("[storage] left_dim in  storage, ", storage.left_dim)
        print("[storage] right_dim in storage, ", storage.right_dim)
        print("lsize = ", lsize)
        print("left_block, right_block = ", left_block.dim, right_block.dim)
        sweep("right", left_block, right_block, interaction, storage)
        plot(left_block.num_sites, right_block.num_sites, "right")  # show geometry
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

            # Rotate all block_operators in right block if needed
            right_block.truncate(right_evecs_truncated)
            right_block.dim = dmax

            truncation_error = 1.0 - sum(right_evals_truncated)
            print("truncation_error = ", truncation_error)
        # Take a picture
        storage.snapshot(left_block, right_block, "right")

    storage.erase("left")
    half_sweeps += 1
    print("--------------This is the end of " + str(half_sweeps) + "th Half-Sweep (R2L)--------------")

evals, evecs = np.linalg.eigh(super_block.block_operators["block_ham"])  # diagonalize the final super_block
print("Eigen values are: ", evals)
print("Ground state energy = ", min(evals))
plot(left_block.num_sites, right_block.num_sites, None)  # show geometry
print("number of left sites = ", left_block.num_sites)
print("number of right sites = ", right_block.num_sites)
