import sys
import src.ParameterEngine as param
from src.Warmup import *
from src.Wavefunction import Wavefunction
from src.helper import Logger, plot


def main(total, shellargs):
    if total != 2:
        print(" ".join(str(x) for x in shellargs))
        raise ValueError('no arguments')
    inputdir = str(shellargs[1])

    # for exporting the logfile
    sys.stdout = Logger()

    num_sites, dmax, interaction, field = param.readinput(inputdir)
    print(" num_sites = ", num_sites, "\n", "#states to keep = ", dmax,
          "\n", "interaction = ", interaction, "\n", "field = ", field, "\n \n")

    # raise invalid input
    if num_sites < 4 or num_sites % 2 != 0:
        raise TypeError("invalid input. Total number of sites must be an even integer and larger than 4")

    # Warm up by infinite size DMRG
    left_block, right_block, storage = warmup(num_sites, dmax, interaction, field)

    num_sweeps = 3  # define the total number of sweeps
    half_sweeps = 0  # for iteration, begin at 0
    super_block = left_block.glue(right_block, interaction)
    print("(fDMRG)left_block.dim before sweep is ", left_block.dim)

    while half_sweeps < 2 * num_sweeps:
        rsize_max = int(right_block.num_sites)
        # left to right
        for rsize in range(rsize_max - 1, 0, -1):
            print("[storage] left_dim in  storage, ", storage.left_dim)
            print("[storage] right_dim in storage, ", storage.right_dim)
            print("rsize = ", rsize)
            sweep("left", left_block, right_block, interaction, field, storage)
            plot(left_block.num_sites, right_block.num_sites, "left")  # show geometry
            print("left_block.dim (The current growing side's dim) = ", left_block.dim)
            super_block = left_block.glue(right_block, interaction)
            evals, evecs = np.linalg.eigh(super_block.block_operators["block_ham"])  # diagonalize the super_block

            # bipartition of the ground state wavefunction
            wf_gs = Wavefunction(left_block.dim, right_block.dim)
            wf_gs.as_matrix = np.reshape(evecs[:, 0], (left_block.dim, right_block.dim))

            if left_block.dim > dmax:
                # Rotate all block_operators in left block if needed
                left_block.truncate(wf_gs, dmax, "left")
                left_block.dim = dmax

            if right_block.dim > dmax:
                # Rotate all block_operators in right block
                right_block.truncate(wf_gs, dmax, "right")
                right_block.dim = dmax

            # Take a picture
            storage.snapshot(left_block, right_block, "left")

        # Erase the right block memory
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
            sweep("right", left_block, right_block, interaction, field, storage)
            plot(left_block.num_sites, right_block.num_sites, "right")  # show geometry
            print("left_block, right_block (After dfmrg) = ", left_block.dim, right_block.dim)
            super_block = left_block.glue(right_block, interaction)
            evals, evecs = np.linalg.eigh(super_block.block_operators["block_ham"])  # diagonalize the super_block

            # bipartition of the ground state wavefunction
            wf_gs = Wavefunction(left_block.dim, right_block.dim)
            wf_gs.as_matrix = np.reshape(evecs[:, 0], (left_block.dim, right_block.dim))

            if left_block.dim > dmax:
                # Rotate all block_operators in left block if needed
                left_block.truncate(wf_gs, dmax, "left")
                left_block.dim = dmax

            if right_block.dim > dmax:
                # Rotate all block_operators in right block if needed
                right_block.truncate(wf_gs, dmax, "right")
                right_block.dim = dmax

            # Take a picture
            storage.snapshot(left_block, right_block, "right")

        # Erase the left block memory
        storage.erase("left")
        half_sweeps += 1
        print("--------------This is the end of " + str(half_sweeps) + "th Half-Sweep (R2L)--------------")

    super_block = left_block.glue(right_block, interaction)
    evals, evecs = np.linalg.eigh(super_block.block_operators["block_ham"])  # diagonalize the final super_block
    print("Eigen values are: ", evals)
    print("Ground state energy = ", min(evals))
    plot(left_block.num_sites, right_block.num_sites, None)  # show geometry
    # print("number of left sites = ", left_block.num_sites)
    # print("number of right sites = ", right_block.num_sites)
    # print("Memory of left block: ", storage.left_operators[0])

    # calculate entanglement spectrum
    wf_gs = Wavefunction(left_block.dim, right_block.dim)
    wf_gs.as_matrix = np.reshape(evecs[:, 0], (left_block.dim, right_block.dim))
    rdm = wf_gs.rdm("right")
    ES_evals, ES_evecs = np.linalg.eigh(rdm)
    ES_evals = ES_evals + 1e-15  # get rid of singularity
    ES = np.around(ES_evals + 1e-15, decimals=6)
    EE = - np.around(np.dot(ES_evals, np.log(ES_evals)), decimals=6)

    print("Entanglement Spectrum: \n", *ES, "\nEntanglement Entropy =", EE)


if __name__ == '__main__':
    shellargs = sys.argv  # get the input argument
    total = len(shellargs)
    main(total, shellargs)
