import numpy as np
from mpi4py import MPI
from math import ceil


class Transposer(object):
    def __init__(self, xpart, ypart, comm=MPI.COMM_WORLD, dtype=None):
        self.comm = comm
        rank = comm.rank
        nranks = comm.size

        assert len(xpart) == nranks
        assert len(ypart) == nranks

        assert len(set(xpart)) == 1
        assert len(set(ypart)) == 1

        self.nx = sum(xpart)
        self.ny = sum(ypart)

        nxlocal = xpart[0]
        nylocal = ypart[0]

        self.nxlocal = nxlocal
        self.nylocal = nylocal

        self.xpart = xpart
        self.ypart = ypart

        self.xshape = tuple((nylocal, self.nx))
        self.yshape = tuple((nxlocal, self.ny))

        self._xshape = tuple((nylocal*self.nx,))
        self._yshape = tuple((nxlocal*self.ny,))

        self._xmsg =  np.zeros(self._xshape, dtype=dtype)
        self._ymsg =  np.zeros(self._yshape, dtype=dtype)

        self.dtype = self._xmsg.dtype

    def forward(self, x, y):
        self._check_arrays(x, y)
        self._packx(x, self._xmsg)
        self.comm.Alltoall(self._xmsg, self._ymsg)
        self._unpacky(self._ymsg, y)

    def backward(self, y, x):
        self._check_arrays(x, y)
        self._packy(y, self._ymsg)
        self.comm.Alltoall(self._ymsg, self._xmsg)
        self._unpackx(self._xmsg, x)

    def _packx(self, x, xm):
        xm.reshape(x.T.shape)[:] = x.T

    def _packy(self, y, ym):
        ym.reshape(y.T.shape)[:] = y.T

    def _unpackx(self, xm, x):
        self._unpack(xm, x, self.nylocal, self.nxlocal)

    def _unpacky(self, ym, y):
        self._unpack(ym, y, self.nxlocal, self.nylocal)

    def _unpack(self, src, dst, srcdim, dstdim):
        buf = src.reshape((self.comm.size, srcdim*dstdim))
        for j in range(srcdim):
            b = j*dstdim
            e = (j+1)*dstdim
            dst[j, :] = buf[:, b:e].reshape(-1)

    def _check_arrays(self, x, y):
        assert x.shape == self.xshape
        assert y.shape == self.yshape
        assert x.dtype == self.dtype
        assert y.dtype == self.dtype
