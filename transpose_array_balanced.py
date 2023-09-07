import numpy as np
from mpi4py import MPI
from transpose import Transposer
from utils import print_in_order, print_once
from time import sleep

comm = MPI.COMM_WORLD
rank = comm.rank

xpart = tuple((2 for _ in range(comm.size)))
ypart = tuple((2 for _ in range(comm.size)))

nx = sum(xpart)
ny = ypart[0]

transpose = Transposer(xpart, ypart, comm, dtype=int)

x = np.zeros(transpose.xshape, dtype=transpose.dtype)
y = np.zeros(transpose.yshape, dtype=transpose.dtype)

for i in range(ny):
    row_length = nx
    rows = i + rank*ny
    offset = 1 + rows*row_length
    x[i, :nx] = offset + np.arange(row_length, dtype=int)

print_once(comm, "original data:")
print_in_order(comm, x)

transpose.forward(x, y)

print_once(comm, "transposed once:")
print_in_order(comm, y)

transpose.backward(y, x)

print_once(comm, "transposed twice:")
print_in_order(comm, x)
