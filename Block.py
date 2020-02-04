import numpy as np
from helper import tensor_prod, truncation, sort


class Block:
    """ A block.
    This class to defines the blocks (one for the left, one for
    the right) needed in the DMRG algorithm. The block comes with
    well defined operators for spin-1/2 site
    """

    def __init__(self, dim, side="left"):

        if dim < 1:
            print("Site dim must be at least 1")
            raise ValueError
        if side not in ["left", "right"]:
            print("Parameter Error: side must be left or right")
            raise ValueError

        self.dim = dim  # size of Hilbert space
        self.num_sites = int(np.log2(self.dim))  # number of sites
        self.side = side  # left by default, can be changed to right
        self.site_operators = {"id": np.eye(2, 2),
                               "s_z": np.array([[0.5, 0], [0, -0.5]]),
                               "s_p": np.array([[0, 1], [0, 0]]),
                               "s_m": np.array([[0, 0], [1, 0]]),
                               "s_x": np.array([[0, 0.5], [0.5, 0]])}
        self.block_operators = {}
        # attributes in a Block object: block_ham: Hamiltonian; block_op = s_z, s_m, s_p: the right end op (2*2)
        # A single site block, so just copy the site operator
        if dim == 2:
            self.block_operators = self.site_operators.copy()
            self.block_operators["block_ham"] = np.zeros((self.dim, self.dim))

        if dim > 2:
            # blocks that have more than 1 site.
            # left and right
            if self.side == "left":
                for op in self.site_operators.keys():
                    self.block_operators[op] = tensor_prod(np.eye(int(self.dim / 2), int(self.dim / 2)),
                                                           self.site_operators[op])
                self.block_operators["block_ham"] = np.zeros((self.dim, self.dim))

            if self.side == "right":
                for op in self.site_operators.keys():
                    self.block_operators[op] = tensor_prod(self.site_operators[op],
                                                           np.eye(int(self.dim / 2), int(self.dim / 2)))
                self.block_operators["block_ham"] = np.zeros((self.dim, self.dim))

    def grow(self, interaction, field=[]):
        """ For growing the left or right block, i.e. to include a new site into the block Hilbert space
        and update all operators of the block

        Parameters:
        -----------
                interaction: a list of N by 2
                field: a list of 3 by 2
                e.g. interaction = [[block_op_1, site_op_1, param],...,[block_op_N, site_op_N, param]]
                     field = [["s_x", h_x], ["s_y", h_y], ["s_z", h_z]]
        """

        if self.side == "left":
            new_bh = tensor_prod(self.block_operators["block_ham"],
                                 np.eye(2, 2))  # extend the old block Hilbert space
        if self.side == "right":
            new_bh = tensor_prod(np.eye(2, 2),
                                 self.block_operators["block_ham"])  # extend the old block Hilbert space

        for i in range(len(interaction)):
            block_op = interaction[i][0]
            site_op = interaction[i][1]
            param = interaction[i][2]
            if self.side == "left":
                new_bh += tensor_prod(self.block_operators[block_op], self.site_operators[site_op]) * param
            if self.side == "right":
                new_bh += tensor_prod(self.site_operators[site_op], self.block_operators[block_op]) * param

        for i in range(len(field)):
            onsite_op = field[i][0]
            param = field[i][1]
            # at chain boundary
            # if self.dim == 2:
            #     new_bh += tensor_prod(self.block_operators[onsite_op], self.site_operators["id"]) * param
            #     new_bh += tensor_prod(self.site_operators["id"], self.block_operators[onsite_op]) * param
            # else:
            if self.side == "left":
                new_bh += tensor_prod(self.block_operators["id"], self.site_operators[onsite_op]) * param
            if self.side == "right":
                new_bh += tensor_prod(self.site_operators[onsite_op], self.block_operators["id"]) * param

        # update attributes
        self.dim *= 2
        self.num_sites += 1
        self.block_operators["block_ham"] = new_bh
        key = list(self.block_operators.keys())
        key.remove("block_ham")
        if self.side == "left":
            for op in key:
                self.block_operators[op] = tensor_prod(np.eye(2, 2), self.block_operators[op])
        else:
            for op in key:
                self.block_operators[op] = tensor_prod(self.block_operators[op], np.eye(2, 2))

    def glue(self, block_B, interaction):
        """ Glue together this block with block B, which is an object in Block class
            connected by interaction matrix
        """
        super_block = Block(self.dim * block_B.dim)  # initially the block hamiltonian is a zero matrix
        if self.side == "left":
            block_A_extended = tensor_prod(self.block_operators["block_ham"], block_B.block_operators["id"])
            block_B_extended = tensor_prod(self.block_operators["id"], block_B.block_operators["block_ham"])
        if self.side == "right":
            block_A_extended = tensor_prod(block_B.block_operators["id"], self.block_operators["block_ham"])
            block_B_extended = tensor_prod(block_B.block_operators["block_ham"], self.block_operators["id"])
        print(self.side, "block_A_extended.shape = ", block_A_extended.shape, "block_B_extended = ",
              block_B_extended.shape)
        super_block.block_operators["block_ham"] = block_A_extended + block_B_extended

        # glue together by the interaction between block_ops
        for i in range(len(interaction)):
            block_A_op = interaction[i][0]
            block_B_op = interaction[i][1]
            param = interaction[i][2]
            if self.side == "left":
                super_block.block_operators["block_ham"] += tensor_prod(self.block_operators[block_A_op],
                                                                        block_B.block_operators[block_B_op]) * param

            if self.side == "right":
                super_block.block_operators["block_ham"] += tensor_prod(block_B.block_operators[block_B_op],
                                                                        self.block_operators[block_A_op]) * param
        return super_block

    def truncate(self, wavefunction, dmax, side):
        """ Truncate (Rotate) all block_operators and the Hamiltonian into the truncated basis
            It is needed in both infinite and finite size DMRG

            Parameters: wavefunction: a Wavefunction object
                        dmax: maximal number of states to keep
                        side: a string, the side attribute of the current block
         """

        print("Truncate the block operators")
        if side not in ["left", "right"]:
            raise TypeError("side must be left or right")

        if side == "left":
            traceout_side = "right"
        else:
            traceout_side = "left"

        rdm = wavefunction.rdm(traceout_side)  # solve reduced density matrix
        evals, evecs = np.linalg.eigh(rdm)
        evals_sorted, evecs_sorted = sort(evals, evecs)
        evecs_truncated = evecs_sorted[:, -dmax:]
        truncation_error = 1.0 - sum(evals_sorted[-dmax:])
        print("truncation_error = ", truncation_error)

        truncation_matrix = evecs_truncated
        for op in self.block_operators.keys():
            self.block_operators[op] = truncation(self.block_operators[op], truncation_matrix)
