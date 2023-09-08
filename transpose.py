import abc
import numpy as np
from mpi4py import MPI


class TransposerInterface(abc.ABC):
    @abc.abstractmethod
    def __init__(self, xpart, ypart, comm=MPI.COMM_WORLD, dtype=None):
        pass

    @abc.abstractmethod
    def forward(self, x, y):
        pass

    @abc.abstractmethod
    def backward(self, y, x):
        pass


class TransposerBase(TransposerInterface):
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

    def forward(self, x, y):
        self._check_arrays(x, y)
        self._packx(x, self._xmsg)
        self.Alltoall(self._xmsg, self._ymsg)
        self._unpacky(self._ymsg, y)

    def backward(self, y, x):
        self._check_arrays(x, y)
        self._packy(y, self._ymsg)
        self.Alltoall(self._ymsg, self._xmsg)
        self._unpackx(self._xmsg, x)

    def _get_buf(self, msg):
        return NotImplementedError

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


class Transposer(TransposerBase):
    def __init__(self, xpart, ypart, comm=MPI.COMM_WORLD, dtype=None):
        super().__init__(xpart, ypart, comm=comm, dtype=dtype)

        assert len(set(xpart)) == 1
        assert len(set(ypart)) == 1

        self._xmsg = self._xbuf
        self._ymsg = self._ybuf

        self.Alltoall = self.comm.Alltoall

    def _get_buf(self, msg):
        return msg


class Transposerv(TransposerBase):
    def __init__(self, xpart, ypart, comm=MPI.COMM_WORLD, dtype=None):
        super().__init__(xpart, ypart, comm=comm, dtype=dtype)

        rank = comm.rank
        nranks = comm.size

        nxlocal = xpart[rank]
        nylocal = ypart[rank]

        self._xcounts = tuple((nylocal*nxremote for nxremote in xpart))
        self._ycounts = tuple((nyremote*nxlocal for nyremote in ypart))

        self._xdispl = tuple((sum(self._xcounts[:r]) for r in range(nranks)))
        self._ydispl = tuple((sum(self._ycounts[:r]) for r in range(nranks)))

        self._xmsg = tuple((self._xbuf, tuple((self._xcounts, self._xdispl))))
        self._ymsg = tuple((self._ybuf, tuple((self._ycounts, self._ydispl))))

        self.Alltoall = self.comm.Alltoallv

    def _get_buf(self, msg):
        return msg[0]


class PaddedTransposer(TransposerInterface):
    def __init__(self, xpart, ypart, comm=MPI.COMM_WORLD, dtype=None):
        self.comm = comm
        rank = comm.rank
        nranks = comm.size

        assert len(xpart) == nranks
        assert len(ypart) == nranks

        assert len(set(ypart)) == 1

        self.xpart = xpart
        self.ypart = ypart

        self.nx = sum(xpart)
        self.ny = sum(ypart)

        nxlocal = xpart[rank]
        nylocal = ypart[rank]

        self.xshape = tuple((nylocal, self.nx))
        self.yshape = tuple((nxlocal, self.ny))

        nxlocal_pad = max(xpart)
        nxpad = comm.size*nxlocal_pad
        msglen = nxlocal_pad*nylocal

        self._xshape = tuple((nylocal*nxpad,))
        self._yshape = tuple((nxlocal_pad*self.ny,))

        self._xpadshape = tuple((nxpad, nylocal))
        self._ypadshape = tuple((comm.size, msglen))

        self._xslice = np.s_[:, :self.nx]
        self._yslice = np.s_[:nxlocal, :]

        self._xbuf = np.zeros(self._xshape, dtype=dtype)
        self._ybuf = np.zeros(self._yshape, dtype=dtype)

        self.dtype = self._xbuf.dtype

        self._xmsg = self._xbuf
        self._ymsg = self._ybuf

        self.Alltoall = self.comm.Alltoall

    def forward(self, x, y):
        self._check_arrays(x, y)
        self._packx(x, self._xmsg)
        # print_once(self.comm, "xmsg:")
        # print_in_order(self.comm, self._xmsg)
        self.Alltoall(self._xmsg, self._ymsg)
        # print_once(self.comm, "ymsg:")
        # print_in_order(self.comm, self._ymsg)
        self._unpacky(self._ymsg, y)

    def backward(self, y, x):
        self._check_arrays(x, y)
        self._packy(y, self._ymsg)
        # print_once(self.comm, "ymsg:")
        # print_in_order(self.comm, self._ymsg)
        self.Alltoall(self._ymsg, self._xmsg)
        # print_once(self.comm, "xmsg:")
        # print_in_order(self.comm, self._xmsg)
        self._unpackx(self._xmsg, x)

    def _get_buf(self, msg):
        buf = msg[0]
        shape = msg[1]
        bufslice = msg[2]
        return buf.reshape(shape)[bufslice]

    def _packx(self, x, xm):
        nx = self.nx
        xmsg = xm.reshape(self._xpadshape).T
        for r in range(self.comm.size):
            xb = sum(self.xpart[:r])
            mlen = self.xpart[r]
            xpad = max(self.xpart)
            yb = r*xpad
            xmsg[:nx, yb:yb+mlen] = x[:, xb:xb+mlen]

    def _unpackx(self, xm, x):
        nx = self.nx
        xmsg = xm.reshape(self._xpadshape).T
        for r in range(self.comm.size):
            xb = sum(self.xpart[:r])
            mlen = self.xpart[r]
            xpad = max(self.xpart)
            yb = r*xpad
            x[:, xb:xb+mlen] = xmsg[:nx, yb:yb+mlen]

    def _packy(self, y, ym):
        rank = self.comm.rank
        xpad = max(self.xpart)
        nylocal = self.ypart[rank]
        nxlocal = self.xpart[rank]
        ybuf = ym.reshape(self._ypadshape)

        for r in range(self.comm.size):
            ymsg = ybuf[r, :].reshape(xpad, nylocal)
            yb = sum(self.ypart[:r])
            yl = nylocal
            ymsg[:nxlocal, :] = y[:, yb:yb+yl]

    def _unpacky(self, ym, y):
        rank = self.comm.rank
        xpad = max(self.xpart)
        nylocal = self.ypart[rank]
        nxlocal = self.xpart[rank]
        ybuf = ym.reshape(self._ypadshape)

        for r in range(self.comm.size):
            ymsg = ybuf[r, :].reshape(xpad, nylocal)
            yb = sum(self.ypart[:r])
            yl = nylocal
            y[:, yb:yb+yl] = ymsg[:nxlocal, :]

    def _check_arrays(self, x, y):
        assert x.shape == self.xshape
        assert y.shape == self.yshape
        assert x.dtype == self.dtype
        assert y.dtype == self.dtype
