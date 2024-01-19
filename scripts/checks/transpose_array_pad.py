from mpi4py import MPI
import numpy as np
from mpitranspose.utils import print_in_order, print_once
from mpitranspose import PaddedTransposer
from time import sleep

comm = MPI.COMM_WORLD
rank = comm.rank
nranks = comm.size

# initialise data

xpart = tuple((4, 3, 1, 2))
ypart = tuple((2 for _ in range(nranks)))

nx = sum(xpart)

assert len(xpart) == len(ypart)

transpose = PaddedTransposer(xpart, ypart, comm=comm, dtype=int)

x = np.zeros(transpose.xshape, dtype=int)
y = np.zeros(transpose.yshape, dtype=int)

for i in range(ypart[rank]):
    offset = 1 + nx*(sum(ypart[:rank]) + i)
    x[i, :] = offset + np.arange(nx, dtype=int)

print_once(comm, "original data:")
print_in_order(comm, x)
sleep(0.05)

transpose.forward(x, y)

print_once(comm, "transposed once:")
print_in_order(comm, y)
sleep(0.05)

transpose.backward(y, x)

print_once(comm, "transposed twice:")
print_in_order(comm, x)
sleep(0.05)
