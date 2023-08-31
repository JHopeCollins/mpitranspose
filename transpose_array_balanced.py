import numpy as np
from mpi4py import MPI
from transpose import Transpose2D
from utils import print_in_order
from time import sleep

comm = MPI.COMM_WORLD
rank = comm.rank

nx = 13
nt = 2

shape = tuple((nt, nx))

transpose = Transpose2D(shape, comm, dtype=int)

x = np.zeros(transpose.xshape, dtype=transpose.dtype)
y = np.zeros(transpose.yshape, dtype=transpose.dtype)

for i in range(nt):
    row_length = nx
    rows = i + rank*nt
    offset = 1 + rows*row_length
    x[i,:nx] = offset + np.arange(row_length, dtype=int)

print_in_order(comm, "process %s original\n%s "
                     % (rank, x))
sleep(0.1)

transpose.forward(x, y)

print_in_order(comm, "process %s transposed once\n%s"
                     % (rank,y.T))
sleep(0.1)

transpose.backward(y, x)

print_in_order(comm, "process %s transposed twice\n%s "
                     % (rank, x))
