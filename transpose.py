import numpy as np
from mpi4py import MPI


class Transposer(object):
    def __init__(self, xpart, ypart, comm=MPI.COMM_WORLD, dtype=None):
        self.comm = comm
        rank = comm.rank
        nranks = comm.size

        assert len(xpart) == nranks
        assert len(ypart) == nranks

        assert len(set(xpart)) == 1
        assert len(set(ypart)) == 1

        self.xpart = xpart
        self.ypart = ypart

        self.nx = sum(xpart)
        self.ny = sum(ypart)

        nxlocal = xpart[rank]
        nylocal = ypart[rank]

        self.xshape = tuple((nylocal, self.nx))
        self.yshape = tuple((nxlocal, self.ny))

        self._xshape = tuple((nylocal*self.nx,))
        self._yshape = tuple((nxlocal*self.ny,))

        self._xbuf = np.zeros(self._xshape, dtype=dtype)
        self._ybuf = np.zeros(self._yshape, dtype=dtype)

        self.dtype = self._xbuf.dtype

        self._xmsg = self._xbuf
        self._ymsg = self._ybuf

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

    def _get_buf(self, msg):
        return msg

    def _packx(self, x, xm):
        xbuf = self._get_buf(xm)
        self._pack(x, xbuf)

    def _packy(self, y, ym):
        ybuf = self._get_buf(ym)
        self._pack(y, ybuf)

    def _unpackx(self, xm, x):
        xbuf = self._get_buf(xm)
        self._unpack(xbuf, x, self.ypart, self.xpart)

    def _unpacky(self, ym, y):
        ybuf = self._get_buf(ym)
        self._unpack(ybuf, y, self.xpart, self.ypart)

    def _pack(self, src, dst):
        dst.reshape(src.T.shape)[:] = src.T

    def _unpack(self, src, dst, srcpart, dstpart):
        rank = self.comm.rank
        for r in range(self.comm.size):
            lsrc = srcpart[rank]
            ldst = dstpart[r]
            b = sum(dstpart[:r])
            e = sum(dstpart[:r+1])
            dstbuf = src[lsrc*b:lsrc*e].reshape((lsrc, ldst))
            dst[:, b:e] = dstbuf

    def _check_arrays(self, x, y):
        assert x.shape == self.xshape
        assert y.shape == self.yshape
        assert x.dtype == self.dtype
        assert y.dtype == self.dtype


class Transposerv(object):
    def __init__(self, xpart, ypart, comm=MPI.COMM_WORLD, dtype=None):
        self.comm = comm
        rank = comm.rank
        nranks = comm.size

        assert len(xpart) == nranks
        assert len(ypart) == nranks

        self.xpart = xpart
        self.ypart = ypart

        self.nx = sum(xpart)
        self.ny = sum(ypart)

        nxlocal = xpart[rank]
        nylocal = ypart[rank]

        self.xshape = tuple((nylocal, self.nx))
        self.yshape = tuple((nxlocal, self.ny))

        self._xshape = tuple((nylocal*self.nx,))
        self._yshape = tuple((nxlocal*self.ny,))

        self._xbuf = np.zeros(self._xshape, dtype=dtype)
        self._ybuf = np.zeros(self._yshape, dtype=dtype)

        self.dtype = self._xbuf.dtype

        self._xcounts = tuple((nylocal*nxremote for nxremote in xpart))
        self._ycounts = tuple((nyremote*nxlocal for nyremote in ypart))

        self._xdispl = tuple((sum(self._xcounts[:i]) for i in range(nranks)))
        self._ydispl = tuple((sum(self._ycounts[:i]) for i in range(nranks)))

        self._xmsg = tuple((self._xbuf, tuple((self._xcounts, self._xdispl))))
        self._ymsg = tuple((self._ybuf, tuple((self._ycounts, self._ydispl))))

    def forward(self, x, y):
        self._check_arrays(x, y)
        self._packx(x, self._xmsg)
        self.comm.Alltoallv(self._xmsg, self._ymsg)
        self._unpacky(self._ymsg, y)

    def backward(self, y, x):
        self._check_arrays(x, y)
        self._packy(y, self._ymsg)
        self.comm.Alltoallv(self._ymsg, self._xmsg)
        self._unpackx(self._xmsg, x)

    def _get_buf(self, msg):
        return msg[0]

    def _packx(self, x, xm):
        xbuf = self._get_buf(xm)
        self._pack(x, xbuf)

    def _packy(self, y, ym):
        ybuf = self._get_buf(ym)
        self._pack(y, ybuf)

    def _unpackx(self, xm, x):
        xbuf = self._get_buf(xm)
        self._unpack(xbuf, x, self.ypart, self.xpart)

    def _unpacky(self, ym, y):
        ybuf = self._get_buf(ym)
        self._unpack(ybuf, y, self.xpart, self.ypart)

    def _pack(self, src, dst):
        dst.reshape(src.T.shape)[:] = src.T

    def _unpack(self, src, dst, srcpart, dstpart):
        rank = self.comm.rank
        for r in range(self.comm.size):
            lsrc = srcpart[rank]
            ldst = dstpart[r]
            b = sum(dstpart[:r])
            e = sum(dstpart[:r+1])
            dstbuf = src[lsrc*b:lsrc*e].reshape((lsrc, ldst))
            dst[:, b:e] = dstbuf

    def _check_arrays(self, x, y):
        assert x.shape == self.xshape
        assert y.shape == self.yshape
        assert x.dtype == self.dtype
        assert y.dtype == self.dtype
