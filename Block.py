import numpy as np
from helper import tensor_prod, truncation


class Block:
    """ A block.
    This class to defines the blocks (one for the left, one for
    the right) needed in the DMRG algorithm. The block comes with
    well defined operators for spin-1/2 site
    """

    def __init__(self, dim, side="left"):

        if dim < 1:
            print("Site dim must be at least 1")
            raise
        if side not in ["left", "right"]:
            print("Parameter Error: side must be left or right")
            raise

        self.dim = dim  # size of Hilbert space
        self.num_sites = int(np.log2(self.dim))  # number of sites
        self.side = side  # left by default, can be changed to right
        self.site_operators = {"id": np.eye(2, 2),
                               "s_z": np.array([[0.5, 0], [0, -0.5]]),
                               "s_p": np.array([[0, 1], [0, 0]]),
                               "s_m": np.array([[0, 0], [1, 0]])}
        # attributes in a Block object: block_ham: Hamiltonian; block_op = s_z, s_m, s_p: the right end op (2*2)
        if dim == 2:
            self.block_operators = {"id": np.eye(self.dim, self.dim),
                                    "block_ham": np.zeros((self.dim, self.dim)),
                                    "s_z": np.array([[0.5, 0], [0, -0.5]]),
                                    "s_p": np.array([[0, 1], [0, 0]]),
                                    "s_m": np.array([[0, 0], [1, 0]])}

        if dim > 2:
            if self.side == "left":
                self.block_operators = {"id": np.eye(self.dim, self.dim),
                                        "block_ham": np.zeros((self.dim, self.dim)),
                                        "s_z": tensor_prod(np.eye(int(self.dim / 2), int(self.dim / 2)),
                                                           np.array([[0.5, 0], [0, -0.5]])),
                                        "s_p": tensor_prod(np.eye(int(self.dim / 2), int(self.dim / 2)),
                                                           np.array([[0, 1], [0, 0]])),
                                        "s_m": tensor_prod(np.eye(int(self.dim / 2), int(self.dim / 2)),
                                                           np.array([[0, 0], [1, 0]]))}

            if self.side == "right":
                self.block_operators = {"id": np.eye(self.dim, self.dim),
                                        "block_ham": np.zeros((self.dim, self.dim)),
                                        "s_z": tensor_prod(np.array([[0.5, 0], [0, -0.5]]),
                                                           np.eye(int(self.dim / 2), int(self.dim / 2))),
                                        "s_p": tensor_prod(np.array([[0, 1], [0, 0]]),
                                                           np.eye(int(self.dim / 2), int(self.dim / 2))),
                                        "s_m": tensor_prod(np.array([[0, 0], [1, 0]]),
                                                           np.eye(int(self.dim / 2), int(self.dim / 2)))}

    def grow(self, interaction):
        """ For growing the left block Hamiltonian, i.e. to include a new site into the block Hilbert space

        Parameters:
        -----------
                interaction: a list of N by 2
                e.g. interaction = [[block_op_1, site_op_1, param],...,[block_op_N, site_op_N, param]]
        """

        if self.dim == 2:
            new_bh = np.zeros((self.dim * 2, self.dim * 2))
        if self.dim > 2:
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

        # update attributes
        self.dim *= 2
        self.num_sites += 1
        self.block_operators["block_ham"] = new_bh
        self.block_operators["id"] = np.eye(self.dim, self.dim)
        if self.side == "left":
            self.block_operators["s_z"] = tensor_prod(np.eye(2, 2), self.block_operators["s_z"])
            self.block_operators["s_m"] = tensor_prod(np.eye(2, 2), self.block_operators["s_m"])
            self.block_operators["s_p"] = tensor_prod(np.eye(2, 2), self.block_operators["s_p"])
        if self.side == "right":
            self.block_operators["s_z"] = tensor_prod(self.block_operators["s_z"], np.eye(2, 2))
            self.block_operators["s_m"] = tensor_prod(self.block_operators["s_m"], np.eye(2, 2))
            self.block_operators["s_p"] = tensor_prod(self.block_operators["s_p"], np.eye(2, 2))

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
            # print(i,self.block_operators[block_A_op].shape,block_B.block_operators[block_B_op].shape)
            if self.side == "left":
                # print("grown but not yet truncated",i,super_block.block_operators["block_ham"].shape,
                # self.block_operators[block_A_op].shape,block_B.block_operators[block_B_op].shape)
                super_block.block_operators["block_ham"] += tensor_prod(self.block_operators[block_A_op],
                                                                        block_B.block_operators[block_B_op]) * param

            if self.side == "right":
                super_block.block_operators["block_ham"] += tensor_prod(block_B.block_operators[block_B_op],
                                                                        self.block_operators[block_A_op]) * param
        return super_block

    def truncate(self, truncation_matrix):
        """ Truncate (Rotate) all block_operators and the Hamiltonian into the truncated basis
            It is needed in both infinite and finite size DMRG
         """
        for op in self.block_operators.keys():
            self.block_operators[op] = truncation(self.block_operators[op], truncation_matrix)
