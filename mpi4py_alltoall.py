from mpi4py import MPI
import numpy as np
from time import sleep
from math import ceil
from utils import print_in_order


def arrstr(array):
    return str(array)
    # return "[" + reduce(add, (f"{i:3}" for i in array)) + "]"


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

nx = 7
offset = 1 + nx*rank
data0 = offset + np.arange(nx, dtype=int)

nchunks = ceil(nx/size)
pad = size*nchunks

senddata = np.zeros(pad, dtype=int)
recvdata = np.zeros(pad, dtype=int)

senddata[:nx] = data0[:]

comm.Alltoall(senddata, recvdata)

data1 = np.zeros((nchunks, size), dtype=int)

for i in range(nchunks):
    data1[i, :] = recvdata[i::nchunks]

print_in_order(comm, "process %s original %s "
                     % (rank, arrstr(data0)))

sleep(0.1)

print_in_order(comm, "process %s sending %s receiving %s "
                     % (rank, arrstr(senddata), arrstr(recvdata)))

sleep(0.1)

print_in_order(comm, "process %s transposed\n%s"
                     % (rank, data1))
