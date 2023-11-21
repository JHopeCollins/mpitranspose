import numpy as np
from mpi4py import MPI
from transpose import Transposerv as Transposer
from timer_context import measure_timing
from utils import print_once
from functools import partial

comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

mpiprint = partial(print_once, comm)

xdofs = int(50e3)
ydofs = 1

mpiprint(f"Total dofs = {size*xdofs*ydofs}")

xpart = tuple((xdofs for _ in range(comm.size)))
ypart = tuple((ydofs for _ in range(comm.size)))

nx = sum(xpart)
ny = ypart[0]

transpose = Transposer(xpart, ypart, comm, dtype=int)

x = np.zeros(transpose.xshape, dtype=transpose.dtype)
y = np.zeros(transpose.yshape, dtype=transpose.dtype)

np.random.seed(12345)

x[:] = np.random.rand(*x.shape)

nmeasure = 4
nwarmup = 1

fmeasurements = measure_timing(partial(transpose.forward, x, y),
                               nmeasure=nmeasure, nwarmup=nwarmup)
mean = round(fmeasurements.mean, 4)
std = round(fmeasurements.std, 4)
mpiprint(f"{nmeasure} forward  iterations: mean = {mean}, std = {std}")

bmeasurements = measure_timing(partial(transpose.backward, y, x),
                               nmeasure=nmeasure, nwarmup=nwarmup)
mean = round(bmeasurements.mean, 4)
std = round(bmeasurements.std, 4)
mpiprint(f"{nmeasure} backward iterations: mean = {mean}, std = {std}")
