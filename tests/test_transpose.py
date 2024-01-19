from mpi4py import MPI
import pytest
import numpy as np
from mpitranspose import Transposer, Transposerv, PaddedTransposer, Transposerw


def gather_rank_values(val, dtype=None, comm=MPI.COMM_WORLD):
    values = np.zeros(comm.size, dtype=dtype)
    values[comm.rank] = val
    comm.Allgather(MPI.IN_PLACE, values)
    return values


def collective_test(passed, msg="", comm=MPI.COMM_WORLD):
    results = gather_rank_values(passed, dtype=bool, comm=comm)
    for rank, result in enumerate(results):
        assert result, f"rank {rank} failed '{msg}'"


def global_array(xpart, ypart, dtype=int, comm=MPI.COMM_WORLD):
    gnx = sum(xpart)
    gny = sum(ypart)
    row_length = gnx

    # global array
    garray = np.zeros((gny, gnx))
    for row in range(gny):
        offset = 1 + row*row_length
        garray[row, :] = offset + np.arange(row_length, dtype=dtype)

    # local sections
    rank = comm.rank
    yb = sum(ypart[:rank])
    ye = sum(ypart[:rank+1])
    xlocal = garray[yb:ye, :]

    xb = sum(xpart[:rank])
    xe = sum(xpart[:rank+1])
    ylocal = garray.T[xb:xe, :]

    return garray, xlocal, ylocal


def transpose_test(xpart, ypart, transpose_type, comm=MPI.COMM_WORLD):
    transposer = transpose_type(xpart, ypart, comm, dtype=int)

    garray, xlocal, ylocal = global_array(xpart, ypart,
                                          dtype=transposer.dtype,
                                          comm=transposer.comm)

    # local arrays
    x = np.zeros(transposer.xshape, dtype=transposer.dtype)
    y = np.zeros(transposer.yshape, dtype=transposer.dtype)

    # initialise values
    x[:] = xlocal[:]

    # transpose once
    transposer.forward(x, y)
    collective_test(np.allclose(y, ylocal),
                    msg="forward transfer")

    # transpose back
    transposer.backward(y, x)
    collective_test(np.allclose(x, xlocal),
                    msg="backward transfer")


@pytest.mark.parallel(nprocs=4)
def test_transpose():
    comm = MPI.COMM_WORLD

    xpart = tuple((3 for _ in range(comm.size)))
    ypart = tuple((2 for _ in range(comm.size)))

    transpose_test(xpart, ypart, Transposer)


@pytest.mark.parallel(nprocs=4)
def test_transposev():
    comm = MPI.COMM_WORLD

    xpart = tuple(range(comm.size))
    ypart = tuple(reversed(range(comm.size)))

    transpose_test(xpart, ypart, Transposerv)


@pytest.mark.parallel(nprocs=4)
def test_transpose_padded():
    comm = MPI.COMM_WORLD

    xpart = tuple((3, 4, 2, 5))
    ypart = tuple(2 for _ in range(comm.size))

    transpose_test(xpart, ypart, PaddedTransposer)


@pytest.mark.parallel(nprocs=4)
def test_transposew():
    comm = MPI.COMM_WORLD

    xpart = tuple((3 for _ in range(comm.size)))
    ypart = tuple((2 for _ in range(comm.size)))

    transpose_test(xpart, ypart, Transposerw)
