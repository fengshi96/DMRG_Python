class Memory:
    """The memory of the system during both infinite and finite size DMRG
        subject to update
    """

    def __init__(self):
        """ block is a Block object"""
        self.sizes = []
        self.left_dim = []
        self.right_dim = []
        self.left_operators = []
        self.right_operators = []

    def snapshot(self, left_block, right_block, bias=None):
        """ copy information in a superblock
            bias tells the function the memory which sub-block to update
            if bias is set to none, update both blocks
        """

        print("copy information from blocks")
        self.sizes.append([left_block.num_sites, right_block.num_sites])
        if bias is None:  # take the global picture
            print("Bias = None is true")
            self.left_dim.append(left_block.dim)
            self.right_dim.append(right_block.dim)
            self.left_operators.append(left_block.block_operators.copy())
            self.right_operators.append(right_block.block_operators.copy())
        elif bias == "left":  # take picture for left block
            print("Bias = left is true")
            self.left_dim.append(left_block.dim)
            self.left_operators.append(left_block.block_operators.copy())
        else:  # take picture for right block
            print("Bias = right is true")
            self.right_dim.append(right_block.dim)
            self.right_operators.append(right_block.block_operators.copy())
