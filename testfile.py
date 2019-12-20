import numpy as np
from helper import tensor_prod

site = u"\u25EF  "
left_size = 5
right_size = 5

left_block = " ".join([site] * left_size)
right_block = " ".join([site] * right_size)

print(left_block + "||  " + right_block)
print(u"\u2192")