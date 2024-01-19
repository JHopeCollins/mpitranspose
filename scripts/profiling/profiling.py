import numpy as np
from functools import partial

from mpi4py import MPI
from mpitranspose import Transposer, Transposerv, Transposerw, PaddedTransposer

from mpitranspose.utils import print_once
from timer_context import measure_timing
from ensemble_comm import EnsembleCommunicator


def profile_transpose(TransposerType, nwarmup, nmeasure,
                      xpart, ypart, comm):
    transpose = TransposerType(xpart, ypart, comm)

    x = np.zeros(transpose.xshape, dtype=transpose.dtype)
    y = np.zeros(transpose.yshape, dtype=transpose.dtype)

    np.random.seed(12345)

    x[:] = np.random.rand(*x.shape)

    fmeasurements = measure_timing(lambda: transpose.forward(x, y),
                                   nmeasure=nmeasure, nwarmup=nwarmup)
    bmeasurements = measure_timing(lambda: transpose.backward(y, x),
                                   nmeasure=nmeasure, nwarmup=nwarmup)

    return fmeasurements, bmeasurements


global_comm = MPI.COMM_WORLD

subsize = 2

ensemble = EnsembleCommunicator(global_comm, subsize)

# spatial and temporal comms
scomm = ensemble.xcomm
tcomm = ensemble.ycomm

# transpose within each temporal comm
comm = tcomm

rank = comm.rank
size = comm.size

mpiprint = partial(print_once, global_comm)

xdofs = int(128e3)
ydofs = 2

nwarmup = 3
nmeasure = 10

mpiprint(f"Number of cores = {global_comm.size}")
mpiprint(f"Number of spatial cores = {scomm.size}")
mpiprint(f"Number of temporal cores = {tcomm.size}")
mpiprint(f"Spatial kDoFs = {xdofs//1000}")
mpiprint(f"Timesteps per core = {ydofs}")
mpiprint(f"kDoFs per core = {xdofs*ydofs//1000}")
mpiprint(f"Total kDoFs = {size*xdofs*ydofs//1000}")
mpiprint("")

xpart = tuple((xdofs for _ in range(comm.size)))
ypart = tuple((ydofs for _ in range(comm.size)))

print_times = ['w']

rnd = partial(round, ndigits=4)

for Type in (Transposer, Transposerv, Transposerw, PaddedTransposer):
    mpiprint("==========")
    mpiprint("")
    mpiprint(f"Transfer type: {Type.__name__}")
    fmeasurements, bmeasurements = \
        profile_transpose(Type, nwarmup, nmeasure, xpart, ypart, comm)

    mean = rnd(fmeasurements.mean)
    std = rnd(fmeasurements.std)
    mpiprint(f"{nmeasure} forward  iterations: mean = {mean}, std = {std}")
    if 'w' in print_times:
        times = [rnd(m) for m in fmeasurements.warmup_times]
        mpiprint(f"warmup times: {times}")
    if 'm' in print_times:
        times = [rnd(m) for m in fmeasurements.measured_times]
        mpiprint(f"measurement times: {times}")
    if len(print_times) > 0:
        mpiprint("")

    mean = rnd(bmeasurements.mean)
    std = rnd(bmeasurements.std)
    mpiprint(f"{nmeasure} backward iterations: mean = {mean}, std = {std}")
    if 'w' in print_times:
        times = [rnd(m) for m in bmeasurements.warmup_times]
        mpiprint(f"warmup times: {times}")
    if 'm' in print_times:
        times = [rnd(m) for m in bmeasurements.measured_times]
        mpiprint(f"measurement times: {times}")
    mpiprint("")
