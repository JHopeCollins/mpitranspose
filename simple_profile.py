import numpy as np
from mpi4py import MPI
from transpose import *
from timer_context import measure_timing
from utils import print_once
from functools import partial


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


comm = MPI.COMM_WORLD
rank = comm.rank
size = comm.size

mpiprint = partial(print_once, comm)

xdofs = int(512e3)
ydofs = 2

nwarmup = 3
nmeasure = 10

mpiprint(f"Number of cores = {size}")
mpiprint(f"Spatial kDoFs = {xdofs//1000}")
mpiprint(f"Timesteps per core = {ydofs}")
mpiprint(f"kDoFs per core = {xdofs*ydofs//1000}")
mpiprint(f"Total kDoFs = {size*xdofs*ydofs//1000}")
mpiprint("")

xpart = tuple((xdofs for _ in range(comm.size)))
ypart = tuple((ydofs for _ in range(comm.size)))

print_times = ['w']

for Type in (Transposer, Transposerv, Transposerw, PaddedTransposer):
    mpiprint("==========")
    mpiprint("")
    mpiprint(f"Transfer type: {Type.__name__}")
    fmeasurements, bmeasurements = \
        profile_transpose(Type, nwarmup, nmeasure, xpart, ypart, comm)

    mean = round(fmeasurements.mean, 4)
    std = round(fmeasurements.std, 4)
    mpiprint(f"{nmeasure} forward  iterations: mean = {mean}, std = {std}")
    if 'w' in print_times:
        times = [round(m, 4) for m in fmeasurements.warmup_times]
        mpiprint(f"warmup times: {times}")
    if 'm' in print_times:
        time = [round(m, 4) for m in fmeasurements.measured_times]
        mpiprint(f"measurement times: {times}")
    if len(print_times) > 0:
        mpiprint("")

    mean = round(bmeasurements.mean, 4)
    std = round(bmeasurements.std, 4)
    mpiprint(f"{nmeasure} backward iterations: mean = {mean}, std = {std}")
    if 'w' in print_times:
        time = [round(m, 4) for m in bmeasurements.warmup_times]
        mpiprint(f"warmup times: {times}")
    if 'm' in print_times:
        time = [round(m, 4) for m in bmeasurements.measured_times]
        mpiprint(f"measurement times: {times}")
    mpiprint("")
