from mpi4py import MPI
import numpy as np
from utils import print_in_order, print_once
from transpose import *
from ensemble_comm import EnsembleCommunicator

def rowcol_comms(part, comm=MPI.COMM_WORLD):
    nslices

global_comm = MPI.COMM_WORLD

# rank/size of global comm
grank = global_comm.rank
gsize = global_comm.size

nslices = 4
slice_length = 1

nx = 64
# rank/size of spatial-axis comm
ssize = gsize // nslices

# spatial / temporal comms
ensemble = EnsembleCommunicator(global_comm, ssize)
scomm = ensemble.xcomm
tcomm = ensemble.ycomm

tsize = tcomm.size
srank = scomm.rank
trank = tcomm.rank

xpart = tuple(nx//tsize for _ in range(tsize))
ypart = tuple(slice_length for _ in range(nslices))

transpose = PaddedTransposer(xpart, ypart, tcomm)

x = np.zeros(transpose.xshape, dtype=transpose.dtype)
y = np.zeros(transpose.yshape, dtype=transpose.dtype)

for i in range(ypart[trank]):
    nx = sum(xpart)
    row_length = nx
    rows = i + trank*ypart[trank]
    offset = 1 + rows*row_length
    x[i, :nx] = offset + np.arange(row_length, dtype=int)

niterations = 8
ftimes = np.zeros(niterations)
btimes = np.zeros(niterations)

for i in range(niterations):
    global_comm.Barrier()

    fstime = MPI.Wtime()
    transpose.forward(x, y)
    fetime = MPI.Wtime()

    global_comm.Barrier()

    bstime = MPI.Wtime()
    transpose.backward(y, x)
    betime = MPI.Wtime()

    ftimes[i] = fetime - fstime
    btimes[i] = betime - bstime

for r in range(ssize):
    global_comm.Barrier()
    if (trank==0) and (srank==r):
        print(f"ftimes = {ftimes}")
        print(f"btimes = {btimes}")
    global_comm.Barrier()
