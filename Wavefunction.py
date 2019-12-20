import numpy as np


class Wavefunction(object):
    """A wavefunction object

    You use this class to represent wavefunctions. Wavefunctions are
    stored as matrices, the rows corresponding to the states of the
    left block, and the columns corresponding to the states of the
    right block.

    """

    def __init__(self, left_dim, right_dim, num_type='double'):
        """Creates an empty wavefunction

        The wavefunction has the correct dimensions, but their
        contents are garbage. You *must* give it a value before use it for
        any calculation.

        Parameters
        ----------
        left_dim : an int
            The dimension of the Hilbert space of the left block
        right_dim : an int
            The dimension of the Hilbert space of the right block
        num_type : a double or complex
            The type of the wavefunction matrix elements.

        """
        super(Wavefunction, self).__init__()
        try:
            self.as_matrix = np.empty((left_dim, right_dim),
                                      num_type)
        except TypeError:
            print("Bad args for wavefunction")
            raise

        self.left_dim = left_dim
        self.right_dim = right_dim
        self.num_type = num_type

    def rdm(self, block_to_be_traced_over):
        """Constructs the reduced DM for this wavefunction.

        You use this function to build the reduced density matrix of this
        wavefunction. The reduced DM is itself a square and hermitian
        matrix as it should.

        Parameters
        ----------
        block_to_be_traced_over : a string
            Which block (left or right) will be traced over.

        Returns
        -------
        result : a numpy array with ndim = 2
            Which is an hermitian matrix with the reduced DM.

        Raises
        ------
            if the name for the block to be traced out is not correct.

        """
        if block_to_be_traced_over not in ('left', 'right'):
            print("block_to_be_traced_over must be left or right")
            raise

        if block_to_be_traced_over == 'left':
            result = np.dot(np.transpose(self.as_matrix), self.as_matrix)
        else:
            result = np.dot(self.as_matrix, np.transpose(self.as_matrix))
        return result
