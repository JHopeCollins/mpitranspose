from contextlib import contextmanager
from mpi4py import MPI


@contextmanager
def mpi_timer(name, stream=print):
    stream(f"{name} starting...")
    stime = MPI.Wtime()
    yield
    etime = MPI.Wtime()
    duration = etime - stime
    stream(f"{name} took {duration} seconds")
