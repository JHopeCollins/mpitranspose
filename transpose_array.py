import numpy as np
from mpi4py import MPI
from transposer import Transposer2D
from utils import print_in_order
from time import sleep

comm = MPI.COMM_WORLD
rank = comm.rank

nx = 13
nt = comm.size

transpose = Transposer2D(nx, comm, dtype=int)

x = np.empty(transpose.xshape, dtype=transpose.dtype)
y = np.empty(transpose.yshape, dtype=transpose.dtype)

offset = 1 + nx*rank
x[:] = offset + np.arange(nx, dtype=int)

transpose.forward(x, y)

print_in_order(comm, "process %s original %s "
                     % (rank, x))
sleep(0.1)

print_in_order(comm, "process %s transposed once\n%s"
                     % (rank,y))
sleep(0.1)

transpose.backward(y, x)

print_in_order(comm, "process %s transposed twice %s "
                     % (rank, x))
