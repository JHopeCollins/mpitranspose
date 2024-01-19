from contextlib import contextmanager
from mpi4py import MPI
from numpy import mean, std
from collections import namedtuple


TimingMeasurements = namedtuple(
    'TimingMeasurements',
    ['nwarmup', 'nmeasure', 'mean', 'std', 'warmup_times', 'measured_times'])


@contextmanager
def mpi_timer(name, stream=print):
    stream(f"{name} starting...")
    stime = MPI.Wtime()
    yield
    etime = MPI.Wtime()
    duration = etime - stime
    stream(f"{name} took {duration} seconds")


def measure_timing(func, nmeasure=3, nwarmup=0, comm=MPI.COMM_WORLD):
    def run():
        comm.Barrier()
        stime = MPI.Wtime()
        func()
        etime = MPI.Wtime()
        return etime - stime

    warmup_times = tuple(run() for _ in range(nwarmup))
    measured_times = tuple(run() for _ in range(nmeasure))

    tmean = mean(measured_times)
    tstd = std(measured_times)

    return TimingMeasurements(nwarmup=nwarmup, nmeasure=nmeasure,
                              mean=tmean, std=tstd,
                              warmup_times=warmup_times,
                              measured_times=measured_times)
