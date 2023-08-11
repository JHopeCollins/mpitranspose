import numpy as np
from mpi4py import MPI
from math import ceil


class Transpose2D(object):
    def __init__(self, local_shape, comm=MPI.COMM_WORLD, dtype=None):
        self.comm = comm
        nranks = comm.size

        nt = local_shape[0]
        nx = local_shape[1]

        nchunk = ceil(nx/nranks)

        if self.comm.rank != self.comm.size-1:
            ny = nchunk
        else:
            ny = max(nx - (nranks-1)*nchunk, 0)

        self.xshape = local_shape
        self.yshape = tuple((ny, nt*nranks))

        pad = nt*nranks*nchunk
        self._msgshape = tuple((pad,))

        self._xmsg = np.zeros(self._msgshape, dtype=dtype)
        self._ymsg = np.zeros(self._msgshape, dtype=dtype)

        self.dtype = self._xmsg.dtype

        self._xshape_pad = tuple((nt, nchunk*nranks))
        self._yshape_pad = tuple((nchunk, nt*nranks))

        self._xbuf = np.zeros(self._xshape_pad, dtype=dtype)

        self._xslice = np.s_[:, :nx]
        self._yslice = np.s_[:ny, :]

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
        xb = self._xbuf
        xb[self._xslice] = x[:]

        nt = self._xshape_pad[0]
        nc = self._yshape_pad[0]
        nr = self.comm.size

        xp = xm.reshape((nr, nt*nc))

        for i in range(nr):
            xp[i, :] = xb[:, i*nc:(i+1)*nc].reshape(-1)

    def _unpackx(self, xm, x):
        xb = self._xbuf
        nr = self.comm.size
        nt = self._xshape_pad[0]
        nc = self._yshape_pad[0]
        for i in range(nr):
            ib = i*nt*nc
            ie = (i+1)*nt*nc
            xr = xm[ib:ie].reshape(nt, nc)

            jb = i*nc
            je = (i+1)*nc
            xb[:, jb:je] = xr[:]

        x[:] = xb[self._xslice]

    def _packy(self, y, ym):
        nc, nw = self._yshape_pad
        ym.reshape(nw, nc).T[self._yslice] = y[:]

    def _unpacky(self, ym, y):
        nc, nw = self._yshape_pad
        y[:] = ym.reshape((nw, nc)).T[self._yslice]

    def _check_arrays(self, x, y):
        assert x.shape == self.xshape
        assert y.shape == self.yshape
        assert x.dtype == self.dtype
        assert y.dtype == self.dtype
