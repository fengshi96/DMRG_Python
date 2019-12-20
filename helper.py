import numpy as np
import sys


def tensor_prod(A, B):
    """Makes the tensor product of two matrices."""

    B_rows = B.shape[0]
    A_rows = A.shape[0]
    B_cols = B.shape[1]
    A_cols = A.shape[1]
    cols = A_cols * B_cols
    rows = A_rows * B_rows

    result = np.empty([rows, cols])
    for i in range(A_rows):
        ii = i * B_rows
        for j in range(A_cols):
            jj = j * B_cols
            result[ii: ii + B_rows, jj: jj + B_cols] = (A[i, j] * B)

    return result


def truncation(matrix_to_transform, transformation_matrix):
    """Transforms a matrix to a new (truncated) basis.

    Parameters
    ----------
    matrix_to_transform : a numpy array of ndim = 2.
        The matrix you want to transform.
    transformation_matrix : a numpy array of ndim = 2.
        The transformation matrix.
    """

    if matrix_to_transform.shape[0] != matrix_to_transform.shape[1]:
        print("Cannot transform a non-square matrix")
        raise
    if matrix_to_transform.shape[1] != transformation_matrix.shape[0]:
        print("Matrix and transformation don't fit")
        raise
    tmp = np.dot(matrix_to_transform, transformation_matrix)
    return np.dot(np.conj(transformation_matrix.transpose()), tmp)


class Logger:
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("logfile.log", "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        pass


def plot(left_size, right_size, growing_side):
    """ For plotting the position of bond that partitions the superblock"""
    if growing_side not in [None, "left", "right"]:
        print("Growing side must be left or right")
        raise
    site = u"\u25EF  "
    rarrow = u"\u2192"
    larrow = u"\u2190"
    left_block = " ".join([site] * left_size)
    right_block = " ".join([site] * right_size)
    if growing_side is None:
        print("Geometry: ", left_block + "||  " + right_block)
    elif growing_side == "left":
        print("Geometry: ", left_block + "||" + rarrow + "  " + right_block)
    else:
        print("Geometry: ", left_block + larrow + "||  " + right_block)
